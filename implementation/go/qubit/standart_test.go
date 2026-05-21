package qubit

import "testing"

func TestGCD(t *testing.T) {
	cases := []struct {
		a, b, want uint64
	}{
		{0, 5, 5},
		{5, 0, 5},
		{12, 8, 4},
		{7, 13, 1},
		{15, 21, 3},
		{100, 75, 25},
	}
	for _, c := range cases {
		got := GCD(c.a, c.b)
		if got != c.want {
			t.Errorf("GCD(%d, %d) = %d, want %d", c.a, c.b, got, c.want)
		}
	}
}
