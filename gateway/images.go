package gateway

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"strings"

	_ "golang.org/x/image/webp"
)

const MaxImageBytes = 256 * 1024
const MaxImageDimension = 2048

func validateImage(value any) (result string, err error) {
	bad := errors.New("invalid image attachment")
	// Decoders process untrusted compressed data. Scope recovery to this boundary;
	// an invalid codec input must never terminate the gateway process.
	defer func() {
		if recover() != nil {
			result = ""
			err = bad
		}
	}()
	s, ok := value.(string)
	if !ok {
		return "", bad
	}
	prefix, encoded, found := strings.Cut(s, ";base64,")
	formats := map[string]string{"data:image/jpeg": "jpeg", "data:image/png": "png", "data:image/webp": "webp"}
	format := formats[prefix]
	if !found || format == "" || len(encoded) == 0 || len(encoded) > ((MaxImageBytes+2)/3)*4 {
		return "", bad
	}
	raw, e := base64.StdEncoding.Strict().DecodeString(encoded)
	if e != nil || len(raw) > MaxImageBytes || base64.StdEncoding.EncodeToString(raw) != encoded {
		return "", bad
	}
	if animatedImage(raw, format) {
		return "", bad
	}
	config, detected, e := image.DecodeConfig(bytes.NewReader(raw))
	if e != nil || detected != format || config.Width < 1 || config.Height < 1 || config.Width > MaxImageDimension || config.Height > MaxImageDimension {
		return "", bad
	}
	_, detected, e = image.Decode(bytes.NewReader(raw))
	if e != nil || detected != format {
		return "", bad
	}
	return s, nil
}
func animatedImage(raw []byte, format string) bool {
	if format == "png" && len(raw) >= 8 {
		hasAnimationControl, hasFrameControl := false, false
		for pos := 8; pos+12 <= len(raw); {
			n := uint64(binary.BigEndian.Uint32(raw[pos : pos+4]))
			if n > uint64(len(raw)-pos-12) {
				return true
			}
			kind := string(raw[pos+4 : pos+8])
			if kind == "acTL" {
				if n != 8 || binary.BigEndian.Uint32(raw[pos+8:pos+12]) > 1 {
					return true
				}
				hasAnimationControl = true
			}
			if kind == "fcTL" {
				hasFrameControl = true
			}
			if kind == "IDAT" && hasAnimationControl && !hasFrameControl {
				return true
			}
			pos += int(n) + 12
		}
	}
	if format == "webp" && len(raw) >= 12 {
		for pos := 12; pos+8 <= len(raw); {
			n := uint64(binary.LittleEndian.Uint32(raw[pos+4 : pos+8]))
			if n > uint64(len(raw)-pos-8) {
				return true
			}
			kind := string(raw[pos : pos+4])
			if kind == "ANIM" || kind == "ANMF" {
				return true
			}
			if kind == "VP8X" && n > 0 && raw[pos+8]&2 != 0 {
				return true
			}
			pos += 8 + int(n) + (int(n) & 1)
		}
	}
	return false
}
