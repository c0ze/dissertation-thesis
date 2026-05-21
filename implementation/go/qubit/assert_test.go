package qubit

import (
	"strings"
	"testing"
)

func TestAssertPasses(t *testing.T) {
	// Should not panic.
	assert(true, "this should not fire")
}

func TestAssertPanicsWithMessage(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic, got none")
		}
		msg := r.(error).Error()
		if !strings.HasPrefix(msg, "qubit: ") {
			t.Fatalf("panic message missing qubit: prefix: %q", msg)
		}
		if !strings.Contains(msg, "bad: 42") {
			t.Fatalf("panic message missing formatted args: %q", msg)
		}
	}()
	assert(false, "bad: %d", 42)
	t.Fatal("unreachable")
}
