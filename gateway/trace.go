package gateway

import (
	"encoding/json"
	"net/http"
	"time"
	"unicode/utf8"
)

// Five fitted requests of up to 640KB, JSON escaping (up to 6x), plus
// bounded model output/evidence. Exceeding this budget is explicit, never truncation.
const maxTraceBytes = 32 * 1024 * 1024
const traceChunkBytes = 4096

type traceState struct {
	sequence, record, bytes         int
	phase                           string
	pending                         []json.RawMessage
	started, ended, limited, failed bool
}

func (s *chatState) traceEvent(name string, v any) {
	if s.trace.failed {
		return
	}
	raw, _ := json.Marshal(v)
	c := http.NewResponseController(s.w)
	_ = c.SetWriteDeadline(time.Now().Add(downstreamWriteTimeout))
	_, err := s.w.Write([]byte("event: " + name + "\ndata: " + string(raw) + "\n\n"))
	if err != nil {
		s.trace.failed = true
		return
	}
	if c.Flush() != nil {
		s.trace.failed = true
	}
}
func (s *chatState) traceEmit(raw json.RawMessage) {
	t := s.trace
	t.record++
	for len(raw) > 0 && !t.failed {
		n := min(traceChunkBytes, len(raw))
		for n < len(raw) && !utf8.RuneStart(raw[n]) {
			n--
		}
		t.sequence++
		s.traceEvent("chat-trace", map[string]any{"version": 1, "requestId": s.diagnostic["requestId"], "sequence": t.sequence, "record": t.record, "data": string(raw[:n]), "last": n == len(raw)})
		raw = raw[n:]
	}
}
func (s *chatState) traceRecord(kind string, payload any) {
	if s.diagnostic == nil {
		return
	}
	if s.trace == nil {
		s.trace = &traceState{phase: "Chat"}
	}
	t := s.trace
	if t.ended || t.limited || t.failed {
		return
	}
	raw, _ := json.Marshal(map[string]any{"phase": t.phase, "kind": kind, "payload": payload})
	if len(raw)+t.bytes > maxTraceBytes {
		t.limited = true
		return
	}
	t.bytes += len(raw)
	if t.started {
		s.traceEmit(raw)
	} else {
		t.pending = append(t.pending, raw)
	}
}
func (s *chatState) traceStart() {
	if s.trace == nil || s.trace.started {
		return
	}
	s.trace.started = true
	for _, raw := range s.trace.pending {
		s.traceEmit(raw)
	}
	s.trace.pending = nil
}
func (s *chatState) traceEnd(complete bool, reason string) {
	if s.trace == nil || s.trace.ended {
		return
	}
	s.traceStart()
	t := s.trace
	if t.limited {
		complete = false
		reason = "trace_limit"
	}
	s.traceEvent("chat-trace-end", map[string]any{"version": 1, "requestId": s.diagnostic["requestId"], "sequence": t.sequence + 1, "records": t.record, "complete": complete, "reason": reason})
	t.ended = true
}
func (s *chatState) traceRequest(phase string, payload map[string]any, settings settings) {
	if s.diagnostic == nil {
		return
	}
	if s.trace == nil {
		s.trace = &traceState{phase: "Chat"}
	}
	s.trace.phase = phase
	// Explicit allowlist excludes credentials, cache salts, and arbitrary fields.
	clean := map[string]any{}
	for _, k := range []string{"model", "messages", "tools", "tool_choice", "parallel_tool_calls", "stream", "stream_options", "max_tokens", "temperature", "chat_template_kwargs"} {
		if v, ok := payload[k]; ok {
			clean[k] = v
		}
	}
	raw, _ := json.Marshal(clean)
	var snapshot map[string]any
	_ = json.Unmarshal(raw, &snapshot)
	if messages, ok := snapshot["messages"].([]any); ok {
		for _, item := range messages {
			m, _ := item.(map[string]any)
			if parts, ok := m["content"].([]any); ok {
				for _, part := range parts {
					p, _ := part.(map[string]any)
					if p["type"] == "image_url" {
						delete(p, "image_url")
						p["placeholder"] = "[binary image omitted]"
					}
				}
			}
		}
	}
	s.traceRecord("request", map[string]any{"sent": snapshot, "effective": map[string]any{"useAdapter": s.adapter, "settings": settings.ints, "reuseCachedPrefixes": settings.reuse}})
}
func (s *chatState) traceResponse(raw any) {
	if s.diagnostic == nil {
		return
	}
	object, _ := raw.(map[string]any)
	clean := map[string]any{}
	if usage, ok := object["usage"].(map[string]any); ok {
		u := map[string]any{}
		for _, k := range []string{"prompt_tokens", "completion_tokens", "total_tokens"} {
			if v, ok := usage[k]; ok {
				u[k] = v
			}
		}
		clean["usage"] = u
	}
	if choices, ok := object["choices"].([]any); ok {
		out := []any{}
		for _, item := range choices {
			c, _ := item.(map[string]any)
			q := map[string]any{}
			for _, k := range []string{"index", "finish_reason"} {
				if v, ok := c[k]; ok {
					q[k] = v
				}
			}
			for _, k := range []string{"message", "delta"} {
				if m, ok := c[k].(map[string]any); ok {
					allowed := map[string]any{}
					for _, key := range []string{"role", "content", "tool_calls"} {
						if v, ok := m[key]; ok {
							if key == "tool_calls" {
								calls := []any{}
								if items, ok := v.([]any); ok {
									for _, item := range items {
										call, _ := item.(map[string]any)
										safe := map[string]any{}
										for _, field := range []string{"id", "index", "type"} {
											if x, ok := call[field]; ok {
												safe[field] = x
											}
										}
										if fn, ok := call["function"].(map[string]any); ok {
											f := map[string]any{}
											for _, field := range []string{"name", "arguments"} {
												if x, ok := fn[field]; ok {
													f[field] = x
												}
											}
											safe["function"] = f
										}
										calls = append(calls, safe)
									}
								}
								allowed[key] = calls
							} else {
								allowed[key] = v
							}
						}
					}
					q[k] = allowed
				}
			}
			out = append(out, q)
		}
		clean["choices"] = out
	}
	if object["error"] != nil {
		clean["failure"] = "upstream_error"
	}
	s.traceRecord("response", clean)
}
func (s *chatState) traceFrame(data []byte) {
	if s.diagnostic == nil {
		return
	}
	var raw any
	if json.Unmarshal(data, &raw) == nil {
		s.traceResponse(raw)
	}
}
