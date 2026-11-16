package gateway

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

func traceRecords(t *testing.T, raw []byte) []map[string]any {
	t.Helper()
	var records []map[string]any
	var pending string
	sequence := 0
	for len(raw) > 0 {
		end := eventBoundary(raw)
		if end < 0 {
			break
		}
		frame := raw[:end]
		raw = raw[end:]
		if !strings.HasPrefix(string(frame), "event: chat-trace\n") {
			continue
		}
		var item map[string]any
		if err := json.Unmarshal(eventData(frame), &item); err != nil {
			t.Fatal(err)
		}
		sequence++
		if item["sequence"] != float64(sequence) {
			t.Fatal("trace sequence")
		}
		pending += item["data"].(string)
		if item["last"] == true {
			var record map[string]any
			if err := json.Unmarshal([]byte(pending), &record); err != nil {
				t.Fatal(err)
			}
			records = append(records, record)
			pending = ""
		}
	}
	if pending != "" {
		t.Fatal("incomplete test trace")
	}
	return records
}

func TestTraceActualFittedRequestsAndImmutablePhases(t *testing.T) {
	for _, web := range []bool{false, true} {
		t.Run(map[bool]string{false: "Chat", true: "Web"}[web], func(t *testing.T) {
			c := testConfig()
			if web {
				c = webConfig(&fakeWeb{})
			}
			c.Diagnostics = true
			var sent []map[string]any
			_, server := testGateway(t, c, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/tokenize" && r.URL.Path != "/v1/chat/completions" {
					basicUpstream(w, r)
					return
				}
				var payload map[string]any
				_ = json.NewDecoder(r.Body).Decode(&payload)
				if r.URL.Path == "/tokenize" {
					if !web && len(payload["messages"].([]any)) > 2 {
						io.WriteString(w, `{"count":40900}`)
					} else {
						io.WriteString(w, `{"count":10}`)
					}
					return
				}
				sent = append(sent, payload)
				if payload["stream"] == false {
					if len(sent) == 1 {
						json.NewEncoder(w).Encode(toolFixture("search_web", `{"query":"synthetic public facts"}`, 10))
					} else {
						json.NewEncoder(w).Encode(stopFixture(10))
					}
					return
				}
				io.WriteString(w, finalFixture)
			}))
			body := `{"messages":[{"role":"user","content":"old"},{"role":"assistant","content":"old answer"},{"role":"user","content":"synthetic unicode 雪 🏓"}],"max_tokens":7680,"stream":true}`
			if web {
				body = strings.TrimSuffix(body, "}") + `,"webSearch":{"version":1,"mode":"auto"}}`
			}
			request := authorizedRequest(context.Background(), server.URL, body)
			request.Header.Set("X-Chat-Diagnostics", "1")
			response, err := http.DefaultClient.Do(request)
			if err != nil {
				t.Fatal(err)
			}
			defer response.Body.Close()
			raw, _ := io.ReadAll(response.Body)
			records := traceRecords(t, raw)
			var captures []map[string]any
			for _, record := range records {
				if record["kind"] == "request" {
					captures = append(captures, record["payload"].(map[string]any)["sent"].(map[string]any))
				}
			}
			if !reflect.DeepEqual(captures, sent) {
				t.Fatalf("capture differs from actual model requests: captures=%d sent=%d", len(captures), len(sent))
			}
			if !web && len(sent[0]["messages"].([]any)) != 2 {
				t.Fatal("ordinary history was not fitted")
			}
			if web {
				if sent[0]["max_tokens"] != float64(2048) || sent[len(sent)-1]["max_tokens"] != float64(2048) {
					t.Fatal("planner cap/final reserve")
				}
				if !strings.Contains(captures[0]["messages"].([]any)[0].(map[string]any)["content"].(string), "TOOL DECISION PHASE") {
					t.Fatal("planner snapshot mutated")
				}
			}
			if !strings.Contains(string(raw), `"complete":true`) {
				t.Fatal("trace not completed")
			}
		})
	}
}

func TestTraceUnicodeChunksSecretExclusionAndOff(t *testing.T) {
	w := httptest.NewRecorder()
	s := chatState{gateway: &Gateway{}, w: w, diagnostic: map[string]any{"requestId": strings.Repeat("a", 32)}}
	payload := map[string]any{"model": "test", "messages": []Message{{"user", strings.Repeat("雪🏓", 100000)}}, "cache_salt": "SECRET", "max_tokens": 128, "temperature": 0.7, "stream": true}
	s.traceRequest("Chat", payload, settings{})
	s.traceStart()
	s.traceResponse(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": "answer", "reasoning_content": "HIDDEN"}, "finish_reason": "length"}}, "usage": map[string]any{"completion_tokens": 10, "private": "SECRET"}})
	s.traceEnd(true, "")
	records := traceRecords(t, w.Body.Bytes())
	if len(records) != 2 || strings.Contains(w.Body.String(), "SECRET") || strings.Contains(w.Body.String(), "HIDDEN") {
		t.Fatal("trace privacy/records")
	}
	if !reflect.DeepEqual(records[0]["payload"].(map[string]any)["sent"].(map[string]any)["messages"].([]any)[0].(map[string]any)["content"], strings.Repeat("雪🏓", 100000)) {
		t.Fatal("Unicode changed")
	}
	off := chatState{w: httptest.NewRecorder()}
	off.traceRequest("Chat", payload, settings{})
	if off.trace != nil {
		t.Fatal("off captured prompt")
	}
}

func TestTraceUpstreamHTTPFailurePreservesStatus(t *testing.T) {
	w := httptest.NewRecorder()
	s := chatState{gateway: &Gateway{}, w: w, diagnostic: map[string]any{"requestId": strings.Repeat("a", 32)}}
	s.traceRequest("Chat", map[string]any{"model": "test", "messages": []Message{{"user", "synthetic"}}}, settings{})
	s.fail(503, "Model unavailable", "", 0)
	var body map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &body)
	if w.Code != 503 || body["trace"] == nil || strings.Contains(w.Header().Get("X-Chat-Diagnostics"), "synthetic") {
		t.Fatal("failure trace/status")
	}
}

func TestPlannerAggregateReserveAndTruncatedTrace(t *testing.T) {
	for _, truncated := range []bool{false, true} {
		t.Run(map[bool]string{false: "aggregate", true: "truncated"}[truncated], func(t *testing.T) {
			provider := &fakeWeb{}
			c := webConfig(provider)
			c.Diagnostics = true
			var budgets []int
			_, server := testGateway(t, c, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/tokenize" && r.URL.Path != "/v1/chat/completions" {
					basicUpstream(w, r)
					return
				}
				var payload map[string]any
				_ = json.NewDecoder(r.Body).Decode(&payload)
				if r.URL.Path == "/tokenize" {
					io.WriteString(w, `{"count":10}`)
					return
				}
				budget := int(payload["max_tokens"].(float64))
				budgets = append(budgets, budget)
				if payload["stream"] == true {
					io.WriteString(w, finalFixture)
					return
				}
				fixture := toolFixture("search_web", `{"query":"synthetic"}`, budget).(map[string]any)
				if truncated {
					fixture["choices"].([]any)[0].(map[string]any)["finish_reason"] = "length"
				} else if len(budgets) == 3 {
					fixture = toolFixture("read_page", `{"sourceId":"S1"}`, budget).(map[string]any)
				}
				json.NewEncoder(w).Encode(fixture)
			}))
			body := strings.Replace(webBody("auto"), `"max_tokens":128`, `"max_tokens":7680`, 1)
			request := authorizedRequest(context.Background(), server.URL, body)
			request.Header.Set("X-Chat-Diagnostics", "1")
			response, err := http.DefaultClient.Do(request)
			if err != nil {
				t.Fatal(err)
			}
			defer response.Body.Close()
			raw, _ := io.ReadAll(response.Body)
			if truncated {
				if len(budgets) != 1 || provider.searches != 0 || !strings.Contains(string(raw), "invalid_tool_call") || !strings.Contains(string(raw), `\"finish_reason\":\"length\"`) || !strings.Contains(string(raw), `"complete":false`) {
					t.Fatal("truncated planner must be traced and rejected before tools", string(raw))
				}
			} else if !reflect.DeepEqual(budgets, []int{2048, 2048, 1536, 2048}) {
				t.Fatalf("aggregate planner/final budgets: %v", budgets)
			}
		})
	}
}
