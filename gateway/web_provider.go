package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"
	"unicode"
)

// WebProvider never receives user credentials or model messages. Extract is
// called only with a URL from the current run's search registry.
type WebProvider interface {
	Search(context.Context, string) ([]WebResult, error)
	Extract(context.Context, string) (string, error)
}
type WebResult struct{ Title, URL, Content string }
type tavilyProvider struct {
	client        *http.Client
	key, endpoint string
}

func (p *tavilyProvider) call(ctx context.Context, path string, payload any, target any) error {
	ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	raw, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, "POST", p.endpoint+path, bytes.NewReader(raw))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+p.key)
	req.Header.Set("Content-Type", "application/json")
	response, err := p.client.Do(req)
	if err != nil {
		return errors.New("web provider unavailable")
	}
	defer response.Body.Close()
	if response.StatusCode != 200 {
		return errors.New("web provider unavailable")
	}
	raw, err = io.ReadAll(io.LimitReader(response.Body, 1024*1024+1))
	if err != nil || len(raw) > 1024*1024 {
		return errors.New("invalid provider response")
	}
	if json.Unmarshal(raw, target) != nil {
		return errors.New("invalid provider response")
	}
	return nil
}
func (p *tavilyProvider) Search(ctx context.Context, query string) ([]WebResult, error) {
	var response struct {
		Results []struct{ Title, URL, Content string }
	}
	err := p.call(ctx, "/search", map[string]any{"query": query, "search_depth": "basic", "max_results": 5, "include_answer": false, "include_raw_content": false, "include_images": false, "auto_parameters": false}, &response)
	if err != nil {
		return nil, err
	}
	var out []WebResult
	for _, r := range response.Results {
		if len(out) == 5 {
			break
		}
		if u, ok := publicWebURL(r.URL); ok {
			out = append(out, WebResult{r.Title, u, r.Content})
		}
	}
	return out, nil
}
func (p *tavilyProvider) Extract(ctx context.Context, target string) (string, error) {
	var response struct {
		Results []struct {
			URL     string
			Content string `json:"raw_content"`
		}
	}
	if err := p.call(ctx, "/extract", map[string]any{"urls": []string{target}, "extract_depth": "basic", "format": "text", "include_images": false, "timeout": 10}, &response); err != nil {
		return "", err
	}
	for _, r := range response.Results {
		if u, ok := publicWebURL(r.URL); ok && u == target && !blank(r.Content) {
			return r.Content, nil
		}
	}
	return "", errors.New("page unavailable")
}

// The gateway never fetches destination URLs itself. This additionally keeps
// local, credentialed and non-HTTP links out of the registry and Extract API.
func publicWebURL(raw string) (string, bool) {
	if len(raw) > 2048 || strings.ContainsFunc(raw, func(r rune) bool { return unicode.IsSpace(r) || r < 0x20 || r == 0x7f }) {
		return "", false
	}
	u, e := url.Parse(raw)
	if e != nil || (u.Scheme != "https" && u.Scheme != "http") || u.Hostname() == "" || u.User != nil {
		return "", false
	}
	host := strings.ToLower(strings.TrimSuffix(u.Hostname(), "."))
	if host == "localhost" || !strings.Contains(host, ".") || strings.HasSuffix(host, ".localhost") || strings.HasSuffix(host, ".local") || strings.HasSuffix(host, ".internal") {
		return "", false
	}
	if ip := net.ParseIP(host); ip != nil && (!ip.IsGlobalUnicast() || ip.IsPrivate() || ip.IsLoopback() || ip.IsLinkLocalUnicast() || ip.IsUnspecified()) {
		return "", false
	}
	if port := u.Port(); port != "" && port != "443" && port != "80" {
		return "", false
	}
	u.Fragment = ""
	return u.String(), true
}
