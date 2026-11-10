// Package gateway implements the authenticated, bounded private chat ingress.
package gateway

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const upstreamJSONLimit = 4 * 1024 * 1024
const downstreamWriteTimeout = 10 * time.Second

var userID = regexp.MustCompile(`^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$`)

type Gateway struct {
	cfg       Config
	client    *http.Client
	admission *Admission
	web       WebProvider
}

func New(c Config) (*Gateway, error) {
	if len([]rune(c.Token)) < 32 {
		return nil, errors.New("a chat token of at least 32 characters is required")
	}
	if c.BaseModel != "" && c.BaseModel == c.Model {
		return nil, errors.New("base and adapter aliases must be distinct")
	}
	if c.Context < 256 {
		return nil, errors.New("invalid model context")
	}
	u, e := url.Parse(c.Upstream)
	if e != nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") {
		return nil, errors.New("invalid upstream URL")
	}
	a, e := newAdmission(c.Concurrency, c.QueueLimit, c.RateLimit, c.UserConcurrency, c.UserQueueLimit, c.UserRateLimit)
	if e != nil {
		return nil, e
	}
	transport := c.Transport
	if transport == nil {
		transport = &http.Transport{Proxy: http.ProxyFromEnvironment, DialContext: (&net.Dialer{Timeout: 5 * time.Second, KeepAlive: 30 * time.Second}).DialContext, ForceAttemptHTTP2: true, MaxIdleConns: 16, MaxIdleConnsPerHost: 8, MaxConnsPerHost: 12, IdleConnTimeout: 90 * time.Second, TLSHandshakeTimeout: 5 * time.Second, ExpectContinueTimeout: time.Second, DisableCompression: true}
	}
	c.Upstream = strings.TrimRight(c.Upstream, "/")
	client := &http.Client{Transport: transport, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	provider := c.WebProvider
	if provider == nil && c.WebSearchAPIKey != "" {
		provider = &tavilyProvider{client: client, key: c.WebSearchAPIKey, endpoint: "https://api.tavily.com"}
	}
	return &Gateway{cfg: c, client: client, admission: a, web: provider}, nil
}
func (g *Gateway) Close() { g.client.CloseIdleConnections() }
func randomHex(n int) string {
	b := make([]byte, n)
	if _, e := rand.Read(b); e != nil {
		panic("cryptographic random source unavailable")
	}
	return hex.EncodeToString(b)
}
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("Content-Type", "application/json")
	http.NewResponseController(w).SetWriteDeadline(time.Now().Add(downstreamWriteTimeout))
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]string{"error": message})
}
func (g *Gateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if subtle.ConstantTimeCompare([]byte(r.Header.Get("Authorization")), []byte("Bearer "+g.cfg.Token)) != 1 {
		writeError(w, 401, "Unauthorized")
		return
	}
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	switch {
	case r.URL.Path == "/health" && r.Method == http.MethodGet:
		g.health(w, r)
	case r.URL.Path == "/v1/chat/completions" && r.Method == http.MethodPost:
		g.chat(w, r)
	case r.URL.Path == "/health" || r.URL.Path == "/v1/chat/completions":
		if r.URL.Path == "/health" {
			w.Header().Set("Allow", "GET")
		} else {
			w.Header().Set("Allow", "POST")
		}
		writeJSON(w, 405, map[string]string{"detail": "Method Not Allowed"})
	default:
		writeJSON(w, 404, map[string]string{"detail": "Not Found"})
	}
}
func (g *Gateway) request(ctx context.Context, method, path string, payload any) (*http.Response, error) {
	var body io.Reader
	if payload != nil {
		raw, e := json.Marshal(payload)
		if e != nil {
			return nil, e
		}
		body = bytes.NewReader(raw)
	}
	r, e := http.NewRequestWithContext(ctx, method, g.cfg.Upstream+path, body)
	if e != nil {
		return nil, e
	}
	if payload != nil {
		r.Header.Set("Content-Type", "application/json")
	}
	if g.cfg.APIKey != "" {
		r.Header.Set("Authorization", "Bearer "+g.cfg.APIKey)
	}
	return g.client.Do(r)
}
func responseJSON(response *http.Response) (any, error) {
	defer response.Body.Close()
	raw, e := io.ReadAll(io.LimitReader(response.Body, upstreamJSONLimit+1))
	if e != nil {
		return nil, e
	}
	if len(raw) > upstreamJSONLimit {
		return nil, errors.New("upstream JSON too large")
	}
	return decodeJSON(raw)
}
func (g *Gateway) servingContext(parent context.Context, target string) (int, error) {
	ctx, cancel := context.WithTimeout(parent, 5*time.Second)
	defer cancel()
	r, e := g.request(ctx, "GET", "/v1/models", nil)
	if e != nil {
		return 0, e
	}
	if r.StatusCode < 200 || r.StatusCode >= 300 {
		r.Body.Close()
		return 0, errors.New("models unavailable")
	}
	raw, e := responseJSON(r)
	if e != nil {
		return 0, e
	}
	m, ok := raw.(map[string]any)
	if !ok {
		return 0, errors.New("invalid models")
	}
	items, ok := m["data"].([]any)
	if !ok {
		return 0, errors.New("invalid model cards")
	}
	models := make(map[string]map[string]any)
	for _, item := range items {
		card, ok := item.(map[string]any)
		if !ok {
			return 0, errors.New("invalid model card")
		}
		id, _ := card["id"].(string)
		if _, exists := models[id]; !exists {
			models[id] = card
		}
	}
	if g.cfg.BaseModel != "" && models[g.cfg.Model]["parent"] != g.cfg.BaseModel {
		return 0, errors.New("adapter lineage unavailable")
	}
	selected := models[target]
	capacity := selected["max_model_len"]
	if capacity == nil {
		if parent, ok := selected["parent"].(string); ok {
			capacity = models[parent]["max_model_len"]
		}
	}
	n, ok := asInt(capacity)
	if !ok || n < 256 {
		return 0, errors.New("serving context unavailable")
	}
	return min(g.cfg.Context, n), nil
}
func (g *Gateway) health(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 15*time.Second)
	defer cancel()
	healthCtx, healthCancel := context.WithTimeout(ctx, 5*time.Second)
	response, e := g.request(healthCtx, "GET", "/health", nil)
	if e == nil {
		response.Body.Close()
	}
	healthCancel()
	if e != nil || response.StatusCode != 200 {
		writeError(w, 503, "Model unavailable")
		return
	}
	capacity, e := g.servingContext(ctx, g.cfg.Model)
	if e == nil && g.cfg.BaseModel != "" {
		var base int
		base, e = g.servingContext(ctx, g.cfg.BaseModel)
		capacity = min(capacity, base)
	}
	if e != nil {
		writeError(w, 503, "Model unavailable")
		return
	}
	limits := make(map[string]any)
	for k, n := range settingsMax {
		limits[k] = n
	}
	limits["contextTokens"] = min(capacity, 40960)
	limits["reuseCachedPrefixes"] = true
	adapter := g.cfg.BaseModel != ""
	adapterVersion := 0
	if adapter {
		adapterVersion = 1
	}
	diagnosticsVersion := 0
	if g.cfg.Diagnostics {
		diagnosticsVersion = 1
	}
	result := map[string]any{"ready": true, "model": g.cfg.Model, "settingsVersion": 1, "docsModeVersion": 1, "accountQuotaVersion": 1, "accountQuotasEnforced": g.cfg.RequireUserID, "adapterToggleVersion": adapterVersion, "adapterToggleAvailable": adapter, "diagnosticsVersion": diagnosticsVersion, "admission": g.admission.snapshot(), "limits": limits}
	result["webSearchVersion"] = 1
	result["webSearchAvailable"] = g.webReason(g.cfg.Model) == "ready"
	webModels := []any{map[string]any{"model": g.cfg.Model, "useAdapter": g.cfg.BaseModel != "", "available": g.webReason(g.cfg.Model) == "ready", "reason": g.webReason(g.cfg.Model)}}
	if g.cfg.BaseModel != "" {
		webModels = append(webModels, map[string]any{"model": g.cfg.BaseModel, "useAdapter": false, "available": g.webReason(g.cfg.BaseModel) == "ready", "reason": g.webReason(g.cfg.BaseModel)})
	}
	result["webSearch"] = map[string]any{"provider": "tavily", "reason": g.webReason(g.cfg.Model), "models": webModels, "limits": map[string]int{"searches": 2, "pages": 3, "modelCalls": 5, "evidenceTokens": 6000, "pageTokens": 1500, "timeoutSeconds": 240}}
	if g.cfg.ModelProfile == "bf16" && g.cfg.ImageInput {
		result["imageInputVersion"] = 1
		result["imageInput"] = map[string]any{"maxBytes": MaxImageBytes, "maxDimension": MaxImageDimension, "mimeTypes": []string{"image/jpeg", "image/png", "image/webp"}}
	}
	writeJSON(w, 200, result)
}

type chatState struct {
	gateway    *Gateway
	w          http.ResponseWriter
	diagnostic map[string]any
	adapter    bool
	image      bool
	started    time.Time
}

func (s *chatState) headers() {
	if s.diagnostic != nil {
		raw, _ := json.Marshal(s.diagnostic)
		s.w.Header().Set("X-Chat-Diagnostics", string(raw))
	}
	if s.gateway.cfg.BaseModel != "" {
		s.w.Header().Set("X-Chat-Adapter-Enabled", strconv.FormatBool(s.adapter))
	}
}
func (s *chatState) fail(status int, message, rejection string, retry int) {
	s.headers()
	if rejection != "" {
		s.w.Header().Set("X-Chat-Rejection", rejection)
	}
	if retry > 0 {
		s.w.Header().Set("Retry-After", strconv.Itoa(retry))
	}
	writeError(s.w, status, message)
}
func (s *chatState) modelFail(message string) {
	if s.image {
		s.fail(503, "The model could not process the image.", "image_processing", 0)
	} else {
		s.fail(503, message, "", 0)
	}
}
func (g *Gateway) chat(w http.ResponseWriter, r *http.Request) {
	ids := r.Header.Values("X-Chat-User-ID")
	user := "__legacy__"
	if len(ids) == 0 && g.cfg.RequireUserID {
		w.Header().Set("X-Chat-Rejection", "user_identity")
		writeError(w, 400, "A chat user ID is required")
		return
	}
	if len(ids) > 0 {
		if len(ids) != 1 || !userID.MatchString(ids[0]) {
			w.Header().Set("X-Chat-Rejection", "user_identity")
			writeError(w, 400, "Invalid chat user ID")
			return
		}
		user = strings.ToLower(ids[0])
	}
	s := chatState{gateway: g, w: w, adapter: g.cfg.BaseModel != "", started: time.Now()}
	requested := r.Header.Get("X-Chat-Diagnostics") == "1"
	if requested && !g.cfg.Diagnostics {
		writeError(w, 403, "Request diagnostics are disabled")
		return
	}
	if requested {
		s.diagnostic = map[string]any{"version": 1, "requestId": randomHex(16), "model": g.cfg.Model, "inputTokensBefore": nil, "inputTokensAfter": nil, "lastCountedInputTokens": nil, "outputBudget": nil, "safetyReserve": nil, "contextBudget": nil, "gatewayTrimmedMessages": 0, "reuseCachedPrefixes": nil}
	}
	if !strings.HasPrefix(r.Header.Get("Content-Type"), "application/json") {
		s.fail(415, "JSON required", "", 0)
		return
	}
	controller := http.NewResponseController(w)
	_ = controller.SetReadDeadline(time.Now().Add(10 * time.Second))
	raw, e := io.ReadAll(io.LimitReader(r.Body, MaxBody+1))
	_ = controller.SetReadDeadline(time.Time{})
	if len(raw) > MaxBody {
		s.fail(413, "Conversation too large", "", 0)
		return
	}
	if e != nil {
		s.fail(400, "Invalid chat request", "", 0)
		return
	}
	v, e := decodeJSON(raw)
	if e != nil {
		s.fail(400, "Invalid chat request", "", 0)
		return
	}
	value, messages, maximum, temperature, e := validateConversation(v)
	if e != nil {
		s.fail(400, "Invalid chat request", "", 0)
		return
	}
	_, hasImage := value["image"]
	docsRaw, hasDocs := value["docsPassages"]
	var passages []passage
	if hasDocs {
		if hasImage {
			s.fail(400, "Runpod Docs mode requires a text-only request.", "docs_image", 0)
			return
		}
		passages, e = validateDocs(docsRaw)
		if e != nil {
			s.fail(400, "Invalid chat request", "", 0)
			return
		}
	}
	if hasImage && !(g.cfg.ModelProfile == "bf16" && g.cfg.ImageInput) {
		s.fail(400, "This model does not accept images.", "image_unsupported", 0)
		return
	}
	if selected, exists := value["useAdapter"]; exists {
		enabled, ok := selected.(bool)
		if !ok || g.cfg.BaseModel == "" {
			s.fail(400, "Invalid chat request", "", 0)
			return
		}
		s.adapter = enabled
	}
	selectedModel := g.cfg.Model
	if g.cfg.BaseModel != "" && !s.adapter {
		selectedModel = g.cfg.BaseModel
	}
	webMode, e := validateWebSearch(value["webSearch"], value)
	if e != nil {
		s.fail(400, "Invalid web search request", "web_invalid", 0)
		return
	}
	if webMode != "off" {
		if hasImage || hasDocs {
			s.fail(400, "Web search requires a text-only chat without Docs mode.", "web_incompatible", 0)
			return
		}
		if g.webReason(selectedModel) != "ready" {
			s.fail(503, "Web search is unavailable for this model.", "web_unavailable", 0)
			return
		}
	}
	if s.diagnostic != nil {
		s.diagnostic["model"] = selectedModel
	}
	settingsRaw, hasSettings := value["settings"]
	settings, e := validateSettings(settingsRaw, hasSettings, g.cfg.Context)
	if e != nil || (hasSettings && maximum != settings.ints["outputTokens"]) {
		s.fail(400, "Invalid chat request", "", 0)
		return
	}
	settings.ints["outputTokens"] = maximum
	if s.diagnostic != nil {
		for k, v := range map[string]any{"outputBudget": maximum, "safetyReserve": settings.ints["safetyTokens"], "contextBudget": settings.ints["contextTokens"], "reuseCachedPrefixes": settings.reuse} {
			s.diagnostic[k] = v
		}
	}
	if len(raw) > settings.ints["requestBytes"] {
		s.fail(413, "Conversation exceeds request byte limit", "", 0)
		return
	}
	if utf16Len(messages[len(messages)-1].Content.(string)) > settings.ints["inputCharacters"] || len(messages) > settings.ints["historyMessages"] || historyBytes(messages) > settings.ints["historyBytes"] || maximum+settings.ints["safetyTokens"] >= settings.ints["contextTokens"] {
		s.fail(400, "Invalid chat request", "", 0)
		return
	}
	if hasImage && s.adapter {
		s.fail(400, "Turn off the BF16 adapter to attach an image.", "image_adapter", 0)
		return
	}
	system := systemPrompt
	if hasDocs {
		system = docsSystemPrompt
		docsPassageOrder(raw, passages)
		messages[len(messages)-1].Content = docsContent(messages[len(messages)-1].Content.(string), passages)
	}
	timeout := settings.ints["timeoutSeconds"]
	if webMode != "off" {
		timeout = min(timeout, 240)
	}
	ctx, cancel := context.WithTimeout(r.Context(), time.Duration(timeout)*time.Second)
	defer cancel()
	ticket, rejection, retry := g.admission.reserve(user)
	if ticket == nil {
		message := map[string]string{"rate_limit": "Chat rate limit reached; retry shortly", "user_rate_limit": "Account chat rate limit reached; retry shortly", "user_capacity": "Account chat capacity reached; retry shortly", "queue_full": "Chat queue is full; retry shortly"}[rejection]
		s.fail(429, message, rejection, retry)
		return
	}
	defer g.admission.release(ticket)
	select {
	case <-r.Context().Done():
		s.fail(499, "Request cancelled", "", 0)
		return
	case <-ctx.Done():
		s.fail(504, "Request timed out waiting for the model", "queue_timeout", 0)
		return
	case <-ticket.ready:
	}
	if hasImage {
		image, e := validateImage(value["image"])
		if e != nil {
			s.fail(400, "Invalid image attachment.", "image_invalid", 0)
			return
		}
		s.image = true
		messages[len(messages)-1].Content = []any{map[string]any{"type": "text", "text": messages[len(messages)-1].Content}, map[string]any{"type": "image_url", "image_url": map[string]string{"url": image}}}
	}
	capacity, e := g.servingContext(ctx, selectedModel)
	if e != nil {
		s.modelFail("Serving context unavailable")
		return
	}
	if settings.ints["contextTokens"] > capacity {
		s.fail(400, "Context budget exceeds the running model capacity", "", 0)
		return
	}
	contextStarted := time.Now()
	if webMode != "off" {
		s.webChat(ctx, selectedModel, webMode, messages, maximum, temperature, settings)
		return
	}
	trimmed := 0
	var conversation []Message
	for {
		conversation = append([]Message{{"system", system}}, messages...)
		response, e := g.request(ctx, "POST", "/tokenize", map[string]any{"model": selectedModel, "messages": conversation, "add_generation_prompt": true, "chat_template_kwargs": map[string]bool{"enable_thinking": false}})
		if e != nil {
			s.modelFail("Model unavailable")
			return
		}
		if response.StatusCode != 200 {
			response.Body.Close()
			s.modelFail("Model tokenizer unavailable")
			return
		}
		counts, e := responseJSON(response)
		if e != nil {
			s.modelFail("Model unavailable")
			return
		}
		m, ok := counts.(map[string]any)
		count, countOK := asInt(m["count"])
		if !ok || !countOK || count < 0 {
			s.modelFail("Invalid tokenizer response")
			return
		}
		if s.diagnostic != nil {
			if s.diagnostic["inputTokensBefore"] == nil {
				s.diagnostic["inputTokensBefore"] = count
			}
			s.diagnostic["lastCountedInputTokens"] = count
		}
		if count <= settings.ints["contextTokens"]-maximum-settings.ints["safetyTokens"] {
			if s.diagnostic != nil {
				s.diagnostic["inputTokensAfter"] = count
			}
			break
		}
		if len(messages) <= 1 {
			s.fail(400, "Message exceeds model context", "", 0)
			return
		}
		messages = messages[2:]
		trimmed += 2
		if s.diagnostic != nil {
			s.diagnostic["gatewayTrimmedMessages"] = trimmed
		}
	}
	fitMs := float64(time.Since(contextStarted)) / float64(time.Millisecond)
	upstreamStarted := time.Now()
	payload := map[string]any{"model": selectedModel, "messages": conversation, "stream": true, "max_tokens": maximum, "temperature": temperature, "chat_template_kwargs": map[string]bool{"enable_thinking": false}}
	if s.diagnostic != nil {
		payload["stream_options"] = map[string]bool{"include_usage": true}
	}
	if !settings.reuse {
		payload["cache_salt"] = randomHex(32)
	}
	response, e := g.request(ctx, "POST", "/v1/chat/completions", payload)
	if e != nil {
		s.modelFail("Model unavailable")
		return
	}
	defer response.Body.Close()
	if response.StatusCode != 200 {
		s.modelFail("Model unavailable")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("X-Context-Trimmed", strconv.Itoa(trimmed))
	w.Header().Set("X-Accel-Buffering", "no")
	if hasDocs {
		w.Header().Set("X-Chat-Docs-Mode", "1")
	}
	if s.image {
		w.Header().Set("X-Chat-Image-Accepted", "true")
	}
	s.headers()
	_ = controller.SetWriteDeadline(time.Now().Add(downstreamWriteTimeout))
	w.WriteHeader(200)
	if e := controller.Flush(); e != nil {
		return
	}
	s.stream(ctx, response.Body, fitMs, upstreamStarted)
}
