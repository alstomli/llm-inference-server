package gateway

import (
	"bytes"
	"encoding/base64"
	"image"
	"image/jpeg"
	"image/png"
	"testing"
)

func FuzzRequestValidation(f *testing.F) {
	for _, seed := range []string{testRequest, `null`, `{"settings":null}`, `{"messages":[{"role":"user","content":"\ud83d\ude00"}],"max_tokens":128.0,"stream":true}`, `{"docsPassages":[{"heading":"h","passage":"p"}]}`, `{"messages":[{"role":{"nested":true},"content":"x"}],"stream":true}`, `{"x":"\ud800"}`} {
		f.Add([]byte(seed))
	}
	f.Fuzz(func(t *testing.T, raw []byte) {
		if len(raw) > MaxBody {
			return
		}
		value, e := decodeJSON(raw)
		if e != nil {
			return
		}
		_, _, _, _, _ = validateConversation(value)
		_, _ = validateSettings(value, true, 40960)
		_, _ = validateDocs(value)
		if m, ok := value.(map[string]any); ok {
			_, _ = validateDocs(m["docsPassages"])
			_, _ = validateSettings(m["settings"], true, 40960)
		}
	})
}
func FuzzImageDecoder(f *testing.F) {
	for _, encode := range []func(io *bytes.Buffer) error{
		func(out *bytes.Buffer) error { return png.Encode(out, image.NewNRGBA(image.Rect(0, 0, 1, 1))) },
		func(out *bytes.Buffer) error { return jpeg.Encode(out, image.NewRGBA(image.Rect(0, 0, 1, 1)), nil) },
	} {
		var out bytes.Buffer
		if e := encode(&out); e != nil {
			f.Fatal(e)
		}
		f.Add(out.Bytes())
	}
	f.Add([]byte("not an image"))
	f.Add([]byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'})
	f.Add([]byte("RIFF\x04\x00\x00\x00WEBP"))
	f.Add([]byte{0xff, 0xd8, 0xff, 0xd9})
	f.Fuzz(func(t *testing.T, raw []byte) {
		if len(raw) > MaxImageBytes {
			return
		}
		for _, mime := range []string{"image/png", "image/jpeg", "image/webp"} {
			_, _ = validateImage("data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(raw))
		}
	})
}
func FuzzSSEParsing(f *testing.F) {
	for _, seed := range []string{"data: [DONE]\n\n", "data: {}\r\n\r\n", "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n", "data: first\ndata: second\n\n", ": keepalive\n\n", "\r\n\r\n"} {
		f.Add([]byte(seed))
	}
	f.Fuzz(func(t *testing.T, raw []byte) {
		if len(raw) > maxSSEEvent {
			return
		}
		end := eventBoundary(raw)
		if end < 0 {
			return
		}
		if end > len(raw) || end < 2 {
			t.Fatal("invalid frame boundary")
		}
		_ = hasContent(eventData(raw[:end]))
	})
}
