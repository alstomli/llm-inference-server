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
