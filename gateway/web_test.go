package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

type fakeWeb struct {
	searches, pages int
	fail            bool
	onSearch        func(context.Context)
}

func (p *fakeWeb) Search(ctx context.Context, q string) ([]WebResult, error) {
	p.searches++
	if p.onSearch != nil {
		p.onSearch(ctx)
	}
	if p.fail {
		return nil, errors.New("provider secret must not escape")
	}
	return []WebResult{{"", "https://example.com/article", strings.Repeat("e", 2000)}, {"dup", "https://example.com/article", "duplicate"}, {"private", "http://127.0.0.1/", "private"}}, nil
}
func (p *fakeWeb) Extract(ctx context.Context, u string) (string, error) {
	p.pages++
	return strings.Repeat("p", 3000), nil
}
func webConfig(provider WebProvider) Config {
	c := testConfig()
	c.WebSearchEnabled = true
	c.WebSearchModelVerified = true
	c.WebSearchParser = "fixture-parser"
	c.WebSearchVerifiedModel = c.Model
	c.WebProvider = provider
	return c
}
func webBody(mode string) string {
	return strings.TrimSuffix(testRequest, "}") + `,"webSearch":{"version":1,"mode":"` + mode + `"}}`
}
func toolFixture(name, args string, usage int) any {
	return map[string]any{"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": nil, "tool_calls": []any{map[string]any{"id": "call_fixture", "type": "function", "function": map[string]any{"name": name, "arguments": args}}}}, "finish_reason": "tool_calls"}}, "usage": map[string]int{"completion_tokens": usage}}
}
func stopFixture(usage int) any {
	return map[string]any{"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": "discard this private planning text"}, "finish_reason": "stop"}}, "usage": map[string]int{"completion_tokens": usage}}
}

const finalFixture = "data: {\"choices\":[{\"delta\":{\"content\":\"Evidence [S1]\"}}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"

func TestWebSearchGateAndOffRegression(t *testing.T) {
	for _, tc := range []struct {
		mode     string
		enabled  bool
		verified bool
		want     int
	}{{"off", false, false, 200}, {"auto", false, false, 503}, {"required", true, false, 503}, {"bogus", true, true, 400}} {
		c := webConfig(&fakeWeb{})
		c.WebSearchEnabled = tc.enabled
		c.WebSearchModelVerified = tc.verified
		g, err := New(c)
		if err != nil {
			t.Fatal(err)
		}
		g.client.Transport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
			w := httptest.NewRecorder()
			basicUpstream(w, r)
			return w.Result(), nil
		})
		w := httptest.NewRecorder()
		g.ServeHTTP(w, authorizedRequest(context.Background(), "http://gateway", webBody(tc.mode)))
		if w.Code != tc.want {
			t.Fatalf("mode %s: %d %s", tc.mode, w.Code, w.Body.String())
		}
		if tc.mode == "off" && (w.Header().Get("X-Chat-Web-Search") != "" || strings.Contains(w.Body.String(), "web-")) {
			t.Fatal("off changed normal stream")
		}
		g.Close()
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestWebSearchReadFinalProtocolAndBudgets(t *testing.T) {
	provider := &fakeWeb{}
	var mu sync.Mutex
	var inference, plannerUsed, finalMax int
	var fitTools, fitPlain bool
	g, server := testGateway(t, webConfig(provider), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tokenize" && r.URL.Path != "/v1/chat/completions" {
			basicUpstream(w, r)
			return
		}
		var payload map[string]any
		if json.NewDecoder(r.Body).Decode(&payload) != nil {
			t.Error("invalid payload")
			return
		}
		if messages, ok := payload["messages"].([]any); ok {
			// The exact deployed Qwen template rejects late system messages.
			// Exercise both tokenizer and inference payloads against that contract.
			for index, item := range messages {
				message := item.(map[string]any)
				if message["role"] == "system" && index != 0 {
					t.Errorf("Qwen rejects system message at index %d", index)
				}
			}
			if payload["stream"] == true {
				if len(messages) != 6 {
					t.Errorf("finalization changed tool transcript length: %d", len(messages))
				}
				first := messages[0].(map[string]any)
				if !strings.Contains(first["content"].(string), "Tool use is finished") {
					t.Error("final directive missing from initial system")
				}
			}
		}
		mu.Lock()
		defer mu.Unlock()
		if r.URL.Path == "/tokenize" {
			if prompt, ok := payload["prompt"].(string); ok {
				fmt.Fprintf(w, `{"count":%d}`, len([]rune(prompt)))
				return
			}
			if payload["tools"] != nil {
				fitTools = true
			} else {
				fitPlain = true
			}
			io.WriteString(w, `{"count":10}`)
			return
		}
		inference++
		if payload["stream"] == true {
			finalMax = int(payload["max_tokens"].(float64))
			if payload["tools"] != nil {
				t.Error("final tools present")
			}
			io.WriteString(w, finalFixture)
			return
		}
		plannerUsed += 10
		switch inference {
		case 1:
			fixture := toolFixture("search_web", `{"query":"public facts"}`, 10).(map[string]any)
			fixture["choices"].([]any)[0].(map[string]any)["finish_reason"] = "stop"
			json.NewEncoder(w).Encode(fixture)
		case 2:
			json.NewEncoder(w).Encode(toolFixture("read_page", `{"sourceId":"S1"}`, 10))
		default:
			json.NewEncoder(w).Encode(stopFixture(10))
		}
	}))
	response, e := http.DefaultClient.Do(authorizedRequest(context.Background(), server.URL, webBody("required")))
	if e != nil {
		t.Fatal(e)
	}
	defer response.Body.Close()
	raw, _ := io.ReadAll(response.Body)
	text := string(raw)
	if response.StatusCode != 200 || response.Header.Get("X-Chat-Web-Search") != "1" {
		t.Fatalf("%d %s", response.StatusCode, text)
	}
	if !strings.Contains(text, `"outcome":"completed"`) || !strings.Contains(text, `"evidenceKind":"page"`) || !strings.Contains(text, "Evidence [S1]") || strings.Contains(text, "discard this") || strings.Contains(text, strings.Repeat("e", 100)) {
		t.Fatal(text)
	}
	if provider.searches != 1 || provider.pages != 1 || inference != 4 || plannerUsed+finalMax > 128 || !fitTools || !fitPlain {
		t.Fatalf("budget/provider/fit %d %d %d %d %d %v %v", provider.searches, provider.pages, inference, plannerUsed, finalMax, fitTools, fitPlain)
	}
	if strings.Index(text, "event: web-complete") > strings.Index(text, "data: [DONE]") {
		t.Fatal("completion after done")
	}
	sequence := 0
	runID := ""
	for _, frame := range strings.Split(text, "\n\n") {
		if !strings.HasPrefix(frame, "event: web-") {
			continue
		}
		var event map[string]any
		if json.Unmarshal(eventData([]byte(frame)), &event) != nil {
			t.Fatal(frame)
		}
		sequence++
		if int(event["sequence"].(float64)) != sequence {
			t.Fatal(event)
		}
		id := event["runId"].(string)
		if runID != "" && runID != id {
			t.Fatal(event)
		}
		runID = id
	}
	if len(g.admission.snapshot()) == 0 {
		t.Fatal("missing admission snapshot")
	}
}

func TestWebSearchRejectsMalformedToolsMissingUsageAndNoEvidence(t *testing.T) {
	for _, test := range []struct {
		name     string
		fixture  any
		provider *fakeWeb
		reason   string
	}{
		{"unknown source", toolFixture("read_page", `{"sourceId":"S999"}`, 10), &fakeWeb{}, "invalid_tool_call"},
		{"arbitrary URL", toolFixture("read_page", `{"url":"http://localhost/secret"}`, 10), &fakeWeb{}, "invalid_tool_call"},
		{"bad args", toolFixture("search_web", `{"query":"x","url":"y"}`, 10), &fakeWeb{}, "invalid_tool_call"},
		{"usage over budget", toolFixture("search_web", `{"query":"x"}`, 1000), &fakeWeb{}, "invalid_model_usage"},
		{"missing usage", map[string]any{"choices": []any{}}, &fakeWeb{}, "invalid_model_usage"},
		{"provider failed", toolFixture("search_web", `{"query":"x"}`, 10), &fakeWeb{fail: true}, "no_web_evidence"},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, server := testGateway(t, webConfig(test.provider), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/v1/chat/completions" {
					basicUpstream(w, r)
					return
				}
				json.NewEncoder(w).Encode(test.fixture)
			}))
			response, e := http.DefaultClient.Do(authorizedRequest(context.Background(), server.URL, webBody("auto")))
			if e != nil {
				t.Fatal(e)
			}
			defer response.Body.Close()
			raw, _ := io.ReadAll(response.Body)
			if !strings.Contains(string(raw), `"outcome":"failed"`) || !strings.Contains(string(raw), test.reason) || strings.Contains(string(raw), "secret must not escape") {
				t.Fatal(string(raw))
			}
		})
	}
}

func TestWebFinalPrematureEOFAndUnexpectedTools(t *testing.T) {
	for _, stream := range []string{"data: [DONE]\n\n", `data: {"choices":[{"delta":{"tool_calls":[{}]}}]}` + "\n\n", "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n", "event: web-sources\ndata: {\"sources\":[]}\n\n"} {
		w := httptest.NewRecorder()
		run := webRun{s: &chatState{w: w}, ctx: context.Background(), id: "test"}
		run.streamFinal(strings.NewReader(stream))
		if !strings.Contains(w.Body.String(), `"outcome":"failed"`) || strings.Contains(w.Body.String(), `"outcome":"completed"`) {
			t.Fatal(w.Body.String())
		}
	}
}

func TestWebStructuredCallFinishCompatibility(t *testing.T) {
	for _, tc := range []struct {
		name, finish, tool, args string
		calls                    int
		want                     bool
	}{
		{"vllm named stop", "stop", "search_web", `{"query":"example domain"}`, 1, true},
		{"auto tool calls", "tool_calls", "read_page", `{"sourceId":"S1"}`, 1, true},
		{"truncated call", "length", "search_web", `{"query":"example"}`, 1, false},
		{"malformed stop", "stop", "search_web", `{"query":`, 1, false},
		{"extra arguments", "stop", "search_web", `{"query":"example","url":"https://example.com"}`, 1, false},
		{"wrong argument type", "stop", "search_web", `{"query":1}`, 1, false},
		{"unknown function", "stop", "execute_code", `{"query":"example"}`, 1, false},
		{"multiple stop calls", "stop", "search_web", `{"query":"example"}`, 2, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := toolFixture(tc.tool, tc.args, 27).(map[string]any)
			choice := raw["choices"].([]any)[0].(map[string]any)
			choice["finish_reason"] = tc.finish
			message := choice["message"].(map[string]any)
			calls := message["tool_calls"].([]any)
			if tc.calls == 2 {
				message["tool_calls"] = append(calls, calls[0])
			}
			_, err := parseWebAssistant(raw)
			if (err == nil) != tc.want {
				t.Fatalf("accepted=%v want=%v error=%v", err == nil, tc.want, err)
			}
		})
	}
	if result, err := parseWebAssistant(stopFixture(10)); err != nil || len(result.ToolCalls) != 0 {
		t.Fatal("normal stop changed", err)
	}
	raw := stopFixture(10).(map[string]any)
	message := raw["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	message["content"] = `{"name":"search_web","arguments":{"query":"example"}}`
	if result, err := parseWebAssistant(raw); err != nil || len(result.ToolCalls) != 0 {
		t.Fatal("raw content authorized a tool", err)
	}
}

func TestWebProviderBoundaryAndCancellation(t *testing.T) {
	for _, u := range []string{"http://localhost/a", "http://127.0.0.1/a", "http://169.254.169.254/a", "https://10.0.0.1/", "file:///etc/passwd", "https://user:pass@example.com/", "https://example.com:1234/"} {
		if _, ok := publicWebURL(u); ok {
			t.Fatal(u)
		}
	}
	if u, ok := publicWebURL("https://example.com/a#fragment"); !ok || u != "https://example.com/a" {
		t.Fatal(u)
	}
	var gotAuth string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		if r.URL.Path == "/search" {
			io.WriteString(w, `{"results":[{"title":"ok","url":"https://example.com/a","content":"ok"}]}`)
			return
		}
		io.WriteString(w, `{"results":[{"url":"https://other.example/a","raw_content":"wrong destination"}],"failed_results":[{"url":"https://example.com/a"}]}`)
	}))
	defer upstream.Close()
	p := tavilyProvider{client: http.DefaultClient, key: "fixture-only", endpoint: upstream.URL}
	results, e := p.Search(context.Background(), "query")
	if e != nil || len(results) != 1 || gotAuth != "Bearer fixture-only" {
		t.Fatal(results, e)
	}
	if _, e = p.Extract(context.Background(), "https://example.com/a"); e == nil {
		t.Fatal("accepted mismatched extract URL")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	start := time.Now()
	if _, e = p.Search(ctx, "x"); e == nil || time.Since(start) > time.Second {
		t.Fatal("cancellation failed")
	}
}

func TestWebToolCapsAndEvidenceLimit(t *testing.T) {
	provider := &fakeWeb{}
	c := webConfig(provider)
	g, e := New(c)
	if e != nil {
		t.Fatal(e)
	}
	defer g.Close()
	g.client.Transport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
		var payload map[string]any
		json.NewDecoder(r.Body).Decode(&payload)
		w := httptest.NewRecorder()
		prompt, _ := payload["prompt"].(string)
		fmt.Fprintf(w, `{"count":%d}`, len([]rune(prompt)))
		return w.Result(), nil
	})
	w := httptest.NewRecorder()
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	run := webRun{s: &chatState{gateway: g, w: w}, ctx: ctx, model: "qwen-bf16", read: map[string]bool{}, workDeadline: time.Now().Add(time.Minute)}
	text, e := run.evidenceText(strings.Repeat("e", 8000), 1500)
	if e != nil || len(text) != 1500 || run.evidence != 1500 {
		t.Fatalf("%d %d %v", len(text), run.evidence, e)
	}
	run.searches = 2
	var call webToolCall
	call.Function.Name = "search_web"
	call.Function.Arguments = `{"query":"x"}`
	if _, e = run.tool(call); e != nil || provider.searches != 0 {
		t.Fatal("search cap failed")
	}
	run.sources = []webSource{{SourceID: "S1", URL: "https://example.com/a"}}
	run.pages = 3
	call.Function.Name = "read_page"
	call.Function.Arguments = `{"sourceId":"S1"}`
	if _, e = run.tool(call); e != nil || provider.pages != 0 {
		t.Fatal("page cap failed")
	}
}

func TestWebCapabilityProfilesAndStrictValidation(t *testing.T) {
	c := webConfig(&fakeWeb{})
	c.Model = "adapter"
	c.BaseModel = "base"
	c.WebSearchVerifiedModel = "base"
	g, e := New(c)
	if e != nil {
		t.Fatal(e)
	}
	defer g.Close()
	if g.webReason("adapter") != "unverified_model" || g.webReason("base") != "ready" {
		t.Fatal("model gate mismatch")
	}
	g.client.Transport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
		w := httptest.NewRecorder()
		if r.URL.Path == "/v1/models" {
			io.WriteString(w, `{"data":[{"id":"adapter","parent":"base","max_model_len":40960},{"id":"base","max_model_len":40960}]}`)
		} else {
			basicUpstream(w, r)
		}
		return w.Result(), nil
	})
	r, _ := http.NewRequest("GET", "http://gateway/health", nil)
	r.Header.Set("Authorization", "Bearer "+c.Token)
	w := httptest.NewRecorder()
	g.ServeHTTP(w, r)
	var result struct {
		WebSearchAvailable bool `json:"webSearchAvailable"`
		WebSearch          struct {
			Models []struct {
				Model      string `json:"model"`
				UseAdapter bool   `json:"useAdapter"`
				Available  bool   `json:"available"`
			} `json:"models"`
		} `json:"webSearch"`
	}
	if json.Unmarshal(w.Body.Bytes(), &result) != nil || result.WebSearchAvailable || len(result.WebSearch.Models) != 2 || result.WebSearch.Models[0].Available || !result.WebSearch.Models[1].Available || result.WebSearch.Models[1].UseAdapter {
		t.Fatal(w.Body.String())
	}
	for _, raw := range []string{`null`, `{"version":2,"mode":"auto"}`, `{"version":1,"mode":"auto","sources":[]}`, `{"mode":"off"}`} {
		var value map[string]any
		decoded, e := decodeJSON([]byte(`{"webSearch":` + raw + `}`))
		if e != nil {
			t.Fatal(e)
		}
		value = decoded.(map[string]any)
		if _, e = validateWebSearch(value["webSearch"], value); e == nil {
			t.Fatal(raw)
		}
	}
}

func TestWebCancellationReleasesProviderAndAdmission(t *testing.T) {
	started := make(chan struct{})
	exited := make(chan struct{})
	provider := &fakeWeb{onSearch: func(ctx context.Context) { close(started); <-ctx.Done(); close(exited) }}
	g, server := testGateway(t, webConfig(provider), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			basicUpstream(w, r)
			return
		}
		json.NewEncoder(w).Encode(toolFixture("search_web", `{"query":"x"}`, 10))
	}))
	ctx, cancel := context.WithCancel(context.Background())
	response, e := http.DefaultClient.Do(authorizedRequest(ctx, server.URL, webBody("required")))
	if e != nil {
		t.Fatal(e)
	}
	defer response.Body.Close()
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("provider not started")
	}
	cancel()
	select {
	case <-exited:
	case <-time.After(2 * time.Second):
		t.Fatal("provider not cancelled")
	}
	deadline := time.Now().Add(time.Second)
	for g.admission.snapshot()["activeRequests"] != 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if g.admission.snapshot()["activeRequests"] != 0 {
		t.Fatal("reservation retained")
	}
}
