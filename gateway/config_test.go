package gateway

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBF16DefaultsAndRejectedProfiles(t *testing.T) {
	defaults := DefaultConfig()
	if defaults.ModelProfile != "bf16" || defaults.Model != "qwen-bf16" {
		t.Fatalf("unexpected default model/profile: %q %q", defaults.Model, defaults.ModelProfile)
	}
	for _, profile := range []string{"4bit", "unknown", ""} {
		t.Run(profile, func(t *testing.T) {
			c := testConfig()
			c.ModelProfile = profile
			g, err := New(c)
			if g != nil {
				g.Close()
			}
			if err == nil {
				t.Fatalf("unsupported profile %q accepted", profile)
			}
			t.Setenv("MODEL_PROFILE", profile)
			t.Setenv("CHAT_TOKEN_FILE", filepath.Join(t.TempDir(), "missing-token"))
			if _, err := ConfigFromEnv(); err == nil || !strings.Contains(err.Error(), "MODEL_PROFILE") {
				t.Fatalf("profile must be rejected before reading tokens or starting services: %v", err)
			}
		})
	}
}

func TestBF16EnvironmentKeepsExplicitBaseAndAdapterAliases(t *testing.T) {
	tokenFile := filepath.Join(t.TempDir(), "token")
	if err := os.WriteFile(tokenFile, []byte(strings.Repeat("t", 48)), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("CHAT_TOKEN_FILE", tokenFile)
	t.Setenv("MODEL_PROFILE", "bf16")
	t.Setenv("SERVED_MODEL_NAME", "custom-bf16-adapter")
	t.Setenv("LORA_ADAPTER_PATH", "fixture-adapter")
	t.Setenv("VLLM_BASE_MODEL_NAME", "custom-bf16-base")
	t.Setenv("CHAT_IMAGE_INPUT_ENABLED", "true")
	c, err := ConfigFromEnv()
	if err != nil {
		t.Fatal(err)
	}
	if c.Model != "custom-bf16-adapter" || c.BaseModel != "custom-bf16-base" || !c.ImageInput {
		t.Fatalf("BF16 adapter/base/image settings lost: %+v", c)
	}
	g, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	g.Close()
}

func TestBoundedCapacityEnvironment(t *testing.T) {
	token := filepath.Join(t.TempDir(), "token")
	if err := os.WriteFile(token, []byte(strings.Repeat("t", 48)), 0600); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"MODEL_PROFILE", "MAX_NUM_SEQS", "CHAT_MAX_CONCURRENT", "CHAT_MAX_WAITING", "CHAT_RATE_LIMIT_PER_MINUTE", "CHAT_USER_MAX_CONCURRENT", "CHAT_USER_MAX_WAITING", "CHAT_USER_RATE_LIMIT_PER_MINUTE"} {
		t.Setenv(key, "")
		os.Unsetenv(key)
	}
	t.Setenv("CHAT_TOKEN_FILE", token)
	cases := []struct {
		name                  string
		values                map[string]string
		active, waiting, rate int
		bad                   bool
	}{
		{"fallback", nil, 2, 6, 20, false},
		{"one engine", map[string]string{"MAX_NUM_SEQS": "1"}, 1, 6, 20, false},
		{"larger engine alone keeps fallback", map[string]string{"MAX_NUM_SEQS": "8"}, 2, 6, 20, false},
		{"production", map[string]string{"MAX_NUM_SEQS": "8", "CHAT_MAX_CONCURRENT": "8", "CHAT_MAX_WAITING": "24", "CHAT_RATE_LIMIT_PER_MINUTE": "80"}, 8, 24, 80, false},
		{"no wait", map[string]string{"CHAT_MAX_WAITING": "0", "CHAT_USER_MAX_WAITING": "0"}, 2, 0, 20, false},
	}
	for _, key := range []string{"MAX_NUM_SEQS", "CHAT_MAX_CONCURRENT", "CHAT_MAX_WAITING", "CHAT_RATE_LIMIT_PER_MINUTE", "CHAT_USER_MAX_CONCURRENT", "CHAT_USER_MAX_WAITING", "CHAT_USER_RATE_LIMIT_PER_MINUTE"} {
		for _, value := range []string{"", "-1", "1.5", "+2", " 2", "1_0", "81"} {
			cases = append(cases, struct {
				name                  string
				values                map[string]string
				active, waiting, rate int
				bad                   bool
			}{key + "=" + value, map[string]string{key: value}, 0, 0, 0, true})
		}
	}
	for _, values := range []map[string]string{
		{"MAX_NUM_SEQS": "0"}, {"MAX_NUM_SEQS": "9"}, {"CHAT_MAX_CONCURRENT": "0"},
		{"MAX_NUM_SEQS": "2", "CHAT_MAX_CONCURRENT": "8"}, {"MAX_NUM_SEQS": "8", "CHAT_MAX_CONCURRENT": "9"},
		{"CHAT_MAX_WAITING": "25"}, {"CHAT_RATE_LIMIT_PER_MINUTE": "0"},
	} {
		cases = append(cases, struct {
			name                  string
			values                map[string]string
			active, waiting, rate int
			bad                   bool
		}{"bounds", values, 0, 0, 0, true})
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for key, value := range tc.values {
				t.Setenv(key, value)
			}
			c, err := ConfigFromEnv()
			if tc.bad {
				if err == nil {
					t.Fatal("invalid capacity accepted")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if c.Concurrency != tc.active || c.QueueLimit != tc.waiting || c.RateLimit != tc.rate {
				t.Fatalf("unexpected limits: %+v", c)
			}
			if c.UserConcurrency != 1 || c.UserRateLimit != 6 || (tc.waiting > 0 && c.UserQueueLimit != 1) {
				t.Fatal("account defaults changed")
			}
		})
	}
}
