package golang

import (
	"testing"

	"github.com/google/uuid"
)

func TestUUid(t *testing.T) {
	s := uuid.NewString()
	t.Log(s)
}
