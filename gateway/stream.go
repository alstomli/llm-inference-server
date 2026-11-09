package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"time"
)

const maxSSEEvent = 1048576
const interruptedStream = "data: {\"error\":{\"message\":\"The model response was interrupted. Please retry.\"}}\n\ndata: [DONE]\n\n"

func (s *chatState) stream(ctx context.Context, body io.Reader, fitMs float64, upstreamStarted time.Time) {
	controller := http.NewResponseController(s.w)
	write := func(data []byte) error {
		if e := controller.SetWriteDeadline(time.Now().Add(downstreamWriteTimeout)); e != nil && !errors.Is(e, http.ErrNotSupported) {
			return e
		}
		if _, e := s.w.Write(data); e != nil {
			return e
		}
		return controller.Flush()
	}
	// The request context closes the upstream transport on disconnect or timeout.
	// Each downstream write has its own deadline, so a slow reader cannot retain
	// an admission slot indefinitely, even if the model is sending continuously.
	chunk := make([]byte, 32*1024)
	if s.diagnostic == nil && !s.image {
		for {
			n, e := body.Read(chunk)
			if n > 0 {
				if write(chunk[:n]) != nil {
					return
				}
			}
			if e != nil {
				if e != io.EOF && ctx.Err() != context.Canceled {
					_ = write([]byte(interruptedStream))
				}
				return
			}
		}
	}
	timing := map[string]any{"contextFitMs": roundMillis(fitMs), "upstreamFirstContentMs": nil, "gatewayFirstContentMs": nil, "gatewayTotalMs": nil}
	timingSent := false
	emitTiming := func() error {
		timing["gatewayTotalMs"] = roundMillis(float64(time.Since(s.started)) / float64(time.Millisecond))
		event, _ := json.Marshal(map[string]any{"version": 1, "requestId": s.diagnostic["requestId"], "timings": timing})
		if e := write(append(append([]byte("event: chat-timing\ndata: "), event...), '\n', '\n')); e != nil {
			return e
		}
		timingSent = true
		return nil
	}
	fail := func() {
		if ctx.Err() == context.Canceled {
			return
		}
		if s.diagnostic != nil && !timingSent {
			if emitTiming() != nil {
				return
			}
		}
		_ = write([]byte(interruptedStream))
	}
	buffer := make([]byte, 0, 32*1024)
	for {
		n, readErr := body.Read(chunk)
		if n > 0 {
			buffer = append(buffer, chunk[:n]...)
			for {
				end := eventBoundary(buffer)
				if end < 0 {
					break
				}
				if end > maxSSEEvent {
					fail()
					return
				}
				frame := buffer[:end]
				data := eventData(frame)
				if s.image && len(data) > 0 && !bytes.Equal(data, []byte("[DONE]")) {
					var item map[string]json.RawMessage
					if e := json.Unmarshal(data, &item); e != nil || item == nil {
						fail()
						return
					}
					if _, exists := item["error"]; exists {
						fail()
						return
					}
				}
				if s.diagnostic != nil && bytes.Equal(data, []byte("[DONE]")) && !timingSent {
					if emitTiming() != nil {
						return
					}
				} else if s.diagnostic != nil && timing["upstreamFirstContentMs"] == nil && hasContent(data) {
					observed := time.Now()
					timing["upstreamFirstContentMs"] = roundMillis(float64(observed.Sub(upstreamStarted)) / float64(time.Millisecond))
					timing["gatewayFirstContentMs"] = roundMillis(float64(observed.Sub(s.started)) / float64(time.Millisecond))
				}
				if write(frame) != nil {
					return
				}
				buffer = buffer[end:]
			}
			if len(buffer) > maxSSEEvent {
				fail()
				return
			}
			if len(buffer) == 0 {
				buffer = buffer[:0]
			} else if cap(buffer) < 32*1024 {
				buffer = append(make([]byte, 0, len(buffer)+32*1024), buffer...)
			}
		}
		if readErr != nil {
			if readErr != io.EOF {
				fail()
				return
			}
			if len(buffer) > 0 {
				if s.image {
					fail()
					return
				}
				if write(append(buffer, '\n', '\n')) != nil {
					return
				}
			}
			if s.diagnostic != nil && !timingSent {
				_ = emitTiming()
			}
			return
		}
	}
}
func roundMillis(v float64) float64 { return math.Round(v*1000) / 1000 }
func eventBoundary(b []byte) int {
	for i := 0; i < len(b)-1; i++ {
		if b[i] != '\n' {
			continue
		}
		if b[i+1] == '\n' {
			return i + 2
		}
		if i+2 < len(b) && b[i+1] == '\r' && b[i+2] == '\n' {
			return i + 3
		}
	}
	return -1
}
func eventData(frame []byte) []byte {
	var parts [][]byte
	for _, line := range bytes.Split(frame, []byte{'\n'}) {
		line = bytes.TrimSuffix(line, []byte{'\r'})
		if bytes.HasPrefix(line, []byte("data:")) {
			parts = append(parts, bytes.TrimLeft(line[5:], " "))
		}
	}
	return bytes.Join(parts, []byte{'\n'})
}
func hasContent(data []byte) bool {
	var item struct {
		Choices []struct {
			Delta struct {
				Content any `json:"content"`
			} `json:"delta"`
		} `json:"choices"`
	}
	if json.Unmarshal(data, &item) != nil || len(item.Choices) == 0 {
		return false
	}
	text, ok := item.Choices[0].Delta.Content.(string)
	return ok && text != ""
}
