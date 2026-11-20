package gateway

import (
	"errors"
	"net/http"
	"os"
	"strconv"
	"strings"
)

// Config describes the private ingress contract. Start with DefaultConfig when
// constructing it directly; ConfigFromEnv preserves the production launcher.
type Config struct {
	Token, Upstream, Model, BaseModel, ModelProfile, APIKey  string
	Context, Concurrency, QueueLimit, RateLimit              int
	UserConcurrency, UserQueueLimit, UserRateLimit           int
	Diagnostics, ImageInput, RequireUserID                   bool
	WebSearchEnabled, WebSearchModelVerified                 bool
	WebSearchAPIKey, WebSearchParser, WebSearchVerifiedModel string
	// WebProvider permits isolated providers in tests; production uses Tavily.
	WebProvider WebProvider
	// Transport permits isolated upstreams in tests and shares one pool in production.
	Transport http.RoundTripper
}

// profile is one named serving configuration: its default alias and whether it has an adapter.
type profile struct {
	alias   string
	adapter bool
}

var profiles = map[string]profile{"bf16": {"qwen-bf16", true}, "4bit": {"qwen-4bit", false}}

func noAdapter(name string) error { return errors.New("the " + name + " profile has no adapter") }

var errProfile = errors.New("MODEL_PROFILE must be bf16 or 4bit")

func DefaultConfig() Config {
	return Config{Upstream: "http://127.0.0.1:8000", Model: "qwen-bf16", ModelProfile: "bf16", Context: 40960, Concurrency: 2, QueueLimit: 6, RateLimit: 20, UserConcurrency: 1, UserQueueLimit: 1, UserRateLimit: 6, RequireUserID: true}
}
func envDefault(k, fallback string) string {
	if v, ok := os.LookupEnv(k); ok {
		return v
	}
	return fallback
}
func ConfigFromEnv() (Config, error) {
	c := DefaultConfig()
	c.ModelProfile = envDefault("MODEL_PROFILE", c.ModelProfile)
	selected, ok := profiles[c.ModelProfile]
	if !ok {
		return c, errProfile
	}
	if !selected.adapter && (os.Getenv("LORA_ADAPTER_PATH") != "" || os.Getenv("VLLM_BASE_MODEL_NAME") != "") {
		return c, noAdapter(c.ModelProfile)
	}
	raw, e := os.ReadFile(envDefault("CHAT_TOKEN_FILE", "/workspace/chat/chat-token"))
	if e != nil {
		return c, errors.New("cannot read chat token file")
	}
	c.Token = strings.TrimSpace(string(raw))
	c.Model = envDefault("SERVED_MODEL_NAME", selected.alias)
	c.Upstream = "http://127.0.0.1:" + envDefault("VLLM_PORT", "8000")
	c.APIKey = os.Getenv("VLLM_API_KEY")
	c.Diagnostics = os.Getenv("CHAT_DIAGNOSTICS_ENABLED") == "true"
	c.ImageInput = os.Getenv("CHAT_IMAGE_INPUT_ENABLED") == "true"
	c.WebSearchEnabled = os.Getenv("CHAT_WEB_SEARCH_ENABLED") == "true"
	c.WebSearchModelVerified = os.Getenv("CHAT_WEB_SEARCH_MODEL_VERIFIED") == "true"
	c.WebSearchAPIKey = os.Getenv("TAVILY_API_KEY")
	c.WebSearchParser = os.Getenv("CHAT_WEB_SEARCH_TOOL_PARSER")
	c.WebSearchVerifiedModel = os.Getenv("CHAT_WEB_SEARCH_VERIFIED_MODEL")
	if os.Getenv("LORA_ADAPTER_PATH") != "" {
		c.BaseModel = os.Getenv("VLLM_BASE_MODEL_NAME")
		if c.BaseModel == "" {
			c.BaseModel = c.Model + "-base"
		}
	}
	required := envDefault("CHAT_REQUIRE_USER_ID", "true")
	if required != "true" && required != "false" {
		return c, errors.New("CHAT_REQUIRE_USER_ID must be true or false")
	}
	c.RequireUserID = required == "true"
	for key, dest := range map[string]*int{"MAX_MODEL_LEN": &c.Context, "CHAT_MAX_CONCURRENT": &c.Concurrency, "CHAT_MAX_WAITING": &c.QueueLimit, "CHAT_RATE_LIMIT_PER_MINUTE": &c.RateLimit, "CHAT_USER_MAX_CONCURRENT": &c.UserConcurrency, "CHAT_USER_MAX_WAITING": &c.UserQueueLimit, "CHAT_USER_RATE_LIMIT_PER_MINUTE": &c.UserRateLimit} {
		if raw, exists := os.LookupEnv(key); exists {
			n, e := strconv.Atoi(raw)
			if e != nil || raw == "" || strings.IndexFunc(raw, func(r rune) bool { return r < '0' || r > '9' }) >= 0 {
				return c, errors.New("invalid integer setting: " + key)
			}
			*dest = n
		}
	}
	engineRaw := envDefault("MAX_NUM_SEQS", "2")
	engine, err := strconv.Atoi(engineRaw)
	if err != nil || engine < 2 || engine > 8 || strings.IndexFunc(engineRaw, func(r rune) bool { return r < '0' || r > '9' }) >= 0 {
		return c, errors.New("MAX_NUM_SEQS must be an integer from 2 through 8")
	}
	// Admission follows the engine unless overridden: N active and 3N waiting.
	if _, explicit := os.LookupEnv("CHAT_MAX_CONCURRENT"); !explicit {
		c.Concurrency = engine
	}
	if _, explicit := os.LookupEnv("CHAT_MAX_WAITING"); !explicit {
		c.QueueLimit = 3 * c.Concurrency
	}
	if c.Concurrency > engine {
		return c, errors.New("CHAT_MAX_CONCURRENT cannot exceed MAX_NUM_SEQS")
	}
	if _, err := newAdmission(c.Concurrency, c.QueueLimit, c.RateLimit, c.UserConcurrency, c.UserQueueLimit, c.UserRateLimit); err != nil {
		return c, err
	}
	return c, nil
}
