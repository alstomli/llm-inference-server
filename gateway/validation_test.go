package gateway

import (
	"strings"
	"testing"
)

func TestValidationIntegerTypesAndUnicode(t *testing.T) {
	for _, n := range []string{"128.0", "1e3", "true", "null"} {
		raw := `{"messages":[{"role":"user","content":"ok"}],"max_tokens":` + n + `,"stream":true}`
		v, e := decodeJSON([]byte(raw))
		if e != nil {
			t.Fatal(e)
		}
		if _, _, _, _, e := validateConversation(v); e == nil {
			t.Fatalf("accepted %s as integer", n)
		}
	}
	if utf16Len("A😀") != 3 {
		t.Fatal("UTF16 character counting")
	}
	if !blank("\x1c\x1f") {
		t.Fatal("Python whitespace")
	}
	for _, raw := range []string{`{"x":"\ud800"}`, `{"x":"\udc00"}`, `{} {}`} {
		if _, e := decodeJSON([]byte(raw)); e == nil {
			t.Fatalf("accepted invalid JSON: %s", raw)
		}
	}
	if _, e := decodeJSON([]byte(`{"x":"\ud83d\ude00","literal":"\\ud800"}`)); e != nil {
		t.Fatal(e)
	}
}
func TestHistoryAndDocumentationEncoding(t *testing.T) {
	messages := []Message{{"user", "<a>\u2028😀\n"}, {"assistant", "ok"}, {"user", "next"}}
	expected := `[{"role":"user","content":"<a>` + "\u2028😀" + `\n"},{"role":"assistant","content":"ok"},{"role":"user","content":"next"}]`
	if historyBytes(messages) != len(expected) {
		t.Fatalf("history bytes %d != %d", historyBytes(messages), len(expected))
	}
	raw := []byte(`{"docsPassages":[{"passage":"p","heading":"h"}]}`)
	passages := []passage{{heading: "h", text: "p"}}
	docsPassageOrder(raw, passages)
	if got := docsContent("q", passages); got != `{"question": "q", "passages": [{"label": "[1]", "passage": "p", "heading": "h"}]}` {
		t.Fatal(got)
	}
}
func TestSettingsRejectUnknownAndExplicitNull(t *testing.T) {
	for _, raw := range []string{`null`, `{"outputTokens":128.0}`, `{"unknown":1}`, `{"reuseCachedPrefixes":1}`} {
		v, _ := decodeJSON([]byte(raw))
		if _, e := validateSettings(v, true, 40960); e == nil {
			t.Fatal("accepted " + raw)
		}
	}
	s, e := validateSettings(nil, false, 4096)
	if e != nil || s.ints["outputTokens"] != 4063 {
		t.Fatalf("adaptive default: %+v %v", s, e)
	}
	if _, e := validateDocs([]any{map[string]any{"heading": "h", "passage": strings.Repeat("😀", 901)}}); e == nil {
		t.Fatal("accepted overlong UTF16 passage")
	}
}
