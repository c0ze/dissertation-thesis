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

func TestModPow(t *testing.T) {
	cases := []struct {
		base, exp, mod, want uint64
	}{
		{2, 10, 1000, 24},
		{7, 0, 15, 1},
		{0, 0, 7, 1},
		{2, 1 << 10, 1000000007, 812734592},    // mod within uint32
		{1<<32 + 1, 5, 1<<33 - 1, 5100273671}, // mod over uint32
	}
	for _, c := range cases {
		got := ModPow(c.base, c.exp, c.mod)
		if got != c.want {
			t.Errorf("ModPow(%d, %d, %d) = %d, want %d",
				c.base, c.exp, c.mod, got, c.want)
		}
	}
}

func TestMulMod(t *testing.T) {
	cases := []struct {
		a, b, mod, want uint64
	}{
		{0, 5, 7, 0},
		{3, 4, 7, 5},        // 12 mod 7 = 5
		{1234567, 7654321, 1000000007, 772047864},
		{1 << 40, 1 << 40, 1 << 50, 0},      // exact-power-of-2 case
	}
	for _, c := range cases {
		got := MulMod(c.a, c.b, c.mod)
		if got != c.want {
			t.Errorf("MulMod(%d, %d, %d) = %d, want %d",
				c.a, c.b, c.mod, got, c.want)
		}
	}
}

func TestMulModZeroModulus(t *testing.T) {
	if got := MulMod(5, 7, 0); got != 0 {
		t.Errorf("MulMod(5, 7, 0) = %d, want 0", got)
	}
}

func TestAddMod(t *testing.T) {
	cases := []struct {
		a, b, mod, want uint64
	}{
		{1, 2, 5, 3},
		{4, 4, 5, 3},          // 8 mod 5 = 3
		{0, 0, 5, 0},
		{1<<63 + 1, 1<<63 + 2, 1<<63 + 5, 1<<63 - 2}, // overflow-prone
		{1<<63, 1<<63 - 1, 1<<63 + 1, 1<<63 - 2},      // mod near 2^63
	}
	for _, c := range cases {
		got := addMod(c.a, c.b, c.mod)
		if got != c.want {
			t.Errorf("addMod(%d, %d, %d) = %d, want %d",
				c.a, c.b, c.mod, got, c.want)
		}
	}
}
