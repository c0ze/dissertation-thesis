package qubit

import (
	"math/cmplx"
	"testing"
)

// assertAmpNear fails the test if |got - want| > AmpTol.
func assertAmpNear(t *testing.T, want, got complex128, name string) {
	t.Helper()
	if cmplx.Abs(want-got) > AmpTol {
		t.Errorf("%s: got %v, want %v (|diff| = %g)",
			name, got, want, cmplx.Abs(want-got))
	}
}
