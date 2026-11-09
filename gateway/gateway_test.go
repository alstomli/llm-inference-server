package gateway

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"image"
	"image/color"
	"image/png"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

const testIdentity = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
const testRequest = `{"messages":[{"role":"user","content":"hello"}],"max_tokens":128,"stream":true}`

func testConfig() Config { c := DefaultConfig(); c.Token = strings.Repeat("t", 32); return c }
func authorizedRequest(ctx context.Context, url, body string) *http.Request {
	r, _ := http.NewRequestWithContext(ctx, "POST", url+"/v1/chat/completions", strings.NewReader(body))
	r.Header.Set("Authorization", "Bearer "+strings.Repeat("t", 32))
	r.Header.Set("X-Chat-User-ID", testIdentity)
	r.Header.Set("Content-Type", "application/json")
	return r
}
func testGateway(t *testing.T, c Config, h http.Handler) (*Gateway, *httptest.Server) {
	t.Helper()
	upstream := httptest.NewServer(h)
	t.Cleanup(upstream.Close)
	c.Upstream = upstream.URL
	g, e := New(c)
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(g.Close)
	server := httptest.NewServer(g)
	t.Cleanup(server.Close)
	return g, server
}
func basicUpstream(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/v1/models":
		io.WriteString(w, `{"data":[{"id":"qwen","max_model_len":40960}]}`)
	case "/tokenize":
		io.WriteString(w, `{"count":10}`)
	case "/health":
		w.WriteHeader(200)
	case "/v1/chat/completions":
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\ndata: [DONE]\n\n")
	default:
		w.WriteHeader(404)
	}
}

func TestGatewayAuthenticationValidationAndBodyBound(t *testing.T) {
	_, server := testGateway(t, testConfig(), http.HandlerFunc(basicUpstream))
	for _, tc := range []struct {
		name, body, identity, token string
		status                      int
		rejection                   string
	}{
		{"unauthorized", testRequest, testIdentity, "wrong", 401, ""},
		{"identity missing", testRequest, "", strings.Repeat("t", 32), 400, "user_identity"},
		{"identity combined", testRequest, testIdentity + ", " + testIdentity, strings.Repeat("t", 32), 400, "user_identity"},
		{"bad settings", strings.TrimSuffix(testRequest, "}") + `,"settings":null}`, testIdentity, strings.Repeat("t", 32), 400, ""},
		{"bounded body", strings.Repeat(" ", MaxBody+1), testIdentity, strings.Repeat("t", 32), 413, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := authorizedRequest(context.Background(), server.URL, tc.body)
			r.Header.Set("Authorization", "Bearer "+tc.token)
			if tc.identity == "" {
				r.Header.Del("X-Chat-User-ID")
			} else {
				r.Header.Set("X-Chat-User-ID", tc.identity)
			}
			response, e := http.DefaultClient.Do(r)
			if e != nil {
				t.Fatal(e)
			}
			defer response.Body.Close()
			if response.StatusCode != tc.status || response.Header.Get("X-Chat-Rejection") != tc.rejection {
				t.Fatalf("status %d header %s", response.StatusCode, response.Header.Get("X-Chat-Rejection"))
			}
			if response.Header.Get("Cache-Control") != "no-store" {
				t.Fatal("missing private cache policy")
			}
		})
	}
}

func TestGatewayTokenFittingDiagnosticsAndDocs(t *testing.T) {
	c := testConfig()
	c.Diagnostics = true
	var mu sync.Mutex
	var tokenConversations, generated [][]Message
	_, server := testGateway(t, c, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tokenize" && r.URL.Path != "/v1/chat/completions" {
			basicUpstream(w, r)
			return
		}
		var payload struct {
			Messages []Message `json:"messages"`
		}
		if e := json.NewDecoder(r.Body).Decode(&payload); e != nil {
			t.Error(e)
		}
		mu.Lock()
		defer mu.Unlock()
		if r.URL.Path == "/tokenize" {
			tokenConversations = append(tokenConversations, payload.Messages)
			if len(payload.Messages) > 2 {
				io.WriteString(w, `{"count":40900}`)
			} else {
				io.WriteString(w, `{"count":10}`)
			}
			return
		}
		generated = append(generated, payload.Messages)
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\r\n\r\ndata: [DONE]\n\n")
	}))
	body := `{"messages":[{"role":"user","content":"old"},{"role":"assistant","content":"old answer"},{"role":"user","content":"q"}],"max_tokens":128,"stream":true,"docsPassages":[{"passage":"evidence","heading":"title"}]}`
	r := authorizedRequest(context.Background(), server.URL, body)
	r.Header.Set("X-Chat-Diagnostics", "1")
	response, e := http.DefaultClient.Do(r)
	if e != nil {
		t.Fatal(e)
	}
	defer response.Body.Close()
	raw, e := io.ReadAll(response.Body)
	if e != nil {
		t.Fatal(e)
	}
	if response.StatusCode != 200 || response.Header.Get("X-Context-Trimmed") != "2" || response.Header.Get("X-Chat-Docs-Mode") != "1" {
		t.Fatalf("response %d %v %s", response.StatusCode, response.Header, raw)
	}
	if bytes.Count(raw, []byte("event: chat-timing")) != 1 || bytes.Index(raw, []byte("event: chat-timing")) > bytes.Index(raw, []byte("data: [DONE]")) {
		t.Fatal("timing event missing or misplaced")
	}
	var diagnostics map[string]any
	if e := json.Unmarshal([]byte(response.Header.Get("X-Chat-Diagnostics")), &diagnostics); e != nil {
		t.Fatal(e)
	}
	if diagnostics["inputTokensBefore"] != float64(40900) || diagnostics["inputTokensAfter"] != float64(10) {
		t.Fatal(diagnostics)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(tokenConversations) != 2 || len(generated) != 1 {
		t.Fatal("context fitting did not retry")
	}
	a, _ := json.Marshal(tokenConversations[1])
	b, _ := json.Marshal(generated[0])
	if !bytes.Equal(a, b) {
		t.Fatal("tokenizer and generation conversation diverged")
	}
	if generated[0][1].Content != `{"question": "q", "passages": [{"label": "[1]", "passage": "evidence", "heading": "title"}]}` {
		t.Fatal(generated[0][1].Content)
	}
}

func TestGatewayDisconnectReleasesUpstreamAndAdmission(t *testing.T) {
	started, cancelled := make(chan struct{}), make(chan struct{})
	g, server := testGateway(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			basicUpstream(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		w.(http.Flusher).Flush()
		close(started)
		<-r.Context().Done()
		close(cancelled)
	}))
	ctx, cancel := context.WithCancel(context.Background())
	response, e := http.DefaultClient.Do(authorizedRequest(ctx, server.URL, testRequest))
	if e != nil {
		t.Fatal(e)
	}
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("model did not start")
	}
	cancel()
	response.Body.Close()
	select {
	case <-cancelled:
	case <-time.After(3 * time.Second):
		t.Fatal("upstream cancellation not propagated")
	}
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if g.admission.snapshot()["activeRequests"] == 0 {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("admission slot leaked")
}

func pngData(t *testing.T, width, height int) []byte {
	t.Helper()
	im := image.NewNRGBA(image.Rect(0, 0, width, height))
	im.Set(0, 0, color.NRGBA{255, 0, 0, 255})
	var out bytes.Buffer
	if e := png.Encode(&out, im); e != nil {
		t.Fatal(e)
	}
	return out.Bytes()
}
func pngChunk(kind string, data []byte) []byte {
	chunk := make([]byte, 12+len(data))
	binary.BigEndian.PutUint32(chunk, uint32(len(data)))
	copy(chunk[4:], kind)
	copy(chunk[8:], data)
	binary.BigEndian.PutUint32(chunk[len(chunk)-4:], crc32.ChecksumIEEE(chunk[4:len(chunk)-4]))
	return chunk
}
func TestImageValidationAndAnimation(t *testing.T) {
	good := pngData(t, 2, 2)
	uri := func(raw []byte) string { return "data:image/png;base64," + base64.StdEncoding.EncodeToString(raw) }
	if _, e := validateImage(uri(good)); e != nil {
		t.Fatal(e)
	}
	control := make([]byte, 8)
	binary.BigEndian.PutUint32(control, 1)
	apngDefault := append(append(append([]byte{}, good[:33]...), pngChunk("acTL", control)...), good[33:]...)
	for name, value := range map[string]any{"wrong MIME": "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(good), "truncated": uri(good[:len(good)/2]), "animated default image": uri(apngDefault), "huge dimensions": uri(pngData(t, 2049, 1)), "newline base64": uri(good) + "\n", "not image": uri([]byte("bad"))} {
		t.Run(name, func(t *testing.T) {
			if _, e := validateImage(value); e == nil {
				t.Fatal("invalid image accepted")
			}
		})
	}
}

func TestImageStreamErrorCannotEchoInput(t *testing.T) {
	c := testConfig()
	c.ModelProfile = "bf16"
	c.ImageInput = true
	_, server := testGateway(t, c, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			basicUpstream(w, r)
			return
		}
		io.WriteString(w, "data: {\"error\":\"secret-image-contents\"}\n\n")
	}))
	request := strings.TrimSuffix(testRequest, "}") + `,"image":"data:image/png;base64,` + base64.StdEncoding.EncodeToString(pngData(t, 1, 1)) + `"}`
	response, e := http.DefaultClient.Do(authorizedRequest(context.Background(), server.URL, request))
	if e != nil {
		t.Fatal(e)
	}
	defer response.Body.Close()
	raw, _ := io.ReadAll(response.Body)
	if bytes.Contains(raw, []byte("secret-image-contents")) || !bytes.Contains(raw, []byte("response was interrupted")) {
		t.Fatalf("unsafe model error %s", raw)
	}
	if response.Header.Get("X-Chat-Image-Accepted") != "true" {
		t.Fatal("image not accepted")
	}
}

func TestBoundedUpstreamModels(t *testing.T) {
	_, server := testGateway(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			io.WriteString(w, strings.Repeat(" ", upstreamJSONLimit+1))
			return
		}
		basicUpstream(w, r)
	}))
	response, e := http.DefaultClient.Do(authorizedRequest(context.Background(), server.URL, testRequest))
	if e != nil {
		t.Fatal(e)
	}
	defer response.Body.Close()
	if response.StatusCode != 503 {
		t.Fatal(response.StatusCode)
	}
}

func TestSlowDownstreamReaderReleasesReservation(t *testing.T) {
	if testing.Short() {
		t.Skip("exercises the production downstream write deadline")
	}
	cancelled := make(chan struct{})
	g, server := testGateway(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			basicUpstream(w, r)
			return
		}
		defer close(cancelled)
		w.Header().Set("Content-Type", "text/event-stream")
		chunk := bytes.Repeat([]byte("x"), 32*1024)
		for {
			select {
			case <-r.Context().Done():
				return
			default:
			}
			if _, e := w.Write(chunk); e != nil {
				return
			}
			if e := http.NewResponseController(w).Flush(); e != nil {
				return
			}
		}
	}))
	address := strings.TrimPrefix(server.URL, "http://")
	remote, e := net.ResolveTCPAddr("tcp", address)
	if e != nil {
		t.Fatal(e)
	}
	conn, e := net.DialTCP("tcp", nil, remote)
	if e != nil {
		t.Fatal(e)
	}
	defer conn.Close()
	if e := conn.SetReadBuffer(1024); e != nil {
		t.Fatal(e)
	}
	_, e = fmt.Fprintf(conn, "POST /v1/chat/completions HTTP/1.1\r\nHost: %s\r\nAuthorization: Bearer %s\r\nX-Chat-User-ID: %s\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s", address, strings.Repeat("t", 32), testIdentity, len(testRequest), testRequest)
	if e != nil {
		t.Fatal(e)
	}
	_ = conn.SetReadDeadline(time.Now().Add(3 * time.Second))
	response, e := http.ReadResponse(bufio.NewReader(conn), nil)
	if e != nil {
		t.Fatal(e)
	}
	if response.StatusCode != 200 {
		t.Fatal(response.StatusCode)
	}
	// Leave both the socket and body open without consuming model bytes.
	select {
	case <-cancelled:
	case <-time.After(downstreamWriteTimeout + 5*time.Second):
		t.Fatal("slow downstream retained the upstream request")
	}
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if g.admission.snapshot()["activeRequests"] == 0 {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("slow downstream leaked admission reservation")
}
