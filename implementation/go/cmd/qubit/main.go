// Command qubit is a small demo binary that exercises the qubit
// library at the algorithm level. Mirrors /c's qubit.c.
package main

import (
	"flag"
	"fmt"
	"math"
	"os"

	"github.com/c0ze/dissertation-thesis/implementation/go/qubit"
)

func main() {
	defer func() {
		if r := recover(); r != nil {
			fmt.Fprintln(os.Stderr, r)
			os.Exit(1)
		}
	}()
	run()
}

func run() {
	algo := flag.String("algo", "bell", "demo to run: bell | qft | grover | shor")
	flag.Parse()
	switch *algo {
	case "bell":
		demoBell()
	case "qft":
		demoQFT()
	case "grover":
		demoGrover()
	case "shor":
		demoShor()
	default:
		fmt.Fprintf(os.Stderr, "unknown algo: %q\n", *algo)
		os.Exit(2)
	}
}

func demoBell() {
	q, err := qubit.NewQreg(2)
	if err != nil {
		panic(err)
	}
	q.InitBasis(0)
	q.ApplyH(0)
	q.ApplyCNOT(0, 1)
	fmt.Printf("Bell |Phi+>: P(00) = %.4f, P(11) = %.4f\n",
		q.ProbOf(0), q.ProbOf(3))
}

func demoQFT() {
	q, err := qubit.NewQreg(4)
	if err != nil {
		panic(err)
	}
	q.InitBasis(0)
	q.ApplyQFT(0, 4)
	fmt.Printf("QFT|0> on 4 qubits: P(0) = %.4f (uniform = %.4f)\n",
		q.ProbOf(0), 1.0/16.0)
}

func demoGrover() {
	// Mark |1111> using ApplyMultiControlledZ as the phase oracle.
	// (For arbitrary marks, the caller would build a custom gate
	// sequence; this demo just shows the canonical |1...1> case.)
	n := 4
	q, err := qubit.NewQreg(n)
	if err != nil {
		panic(err)
	}
	q.InitBasis(0)
	oracle := func(q *qubit.Qreg, _ interface{}) {
		q.ApplyMultiControlledZ(n)
	}
	iters := int(math.Pi / 4 * math.Sqrt(float64(int(1)<<n)))
	q.ApplyGrover(n, oracle, nil, iters)
	fmt.Printf("Grover marked |1111>: P(15) = %.4f\n", q.ProbOf(15))
}

func demoShor() {
	res := qubit.ShorFactor(15, 8)
	fmt.Printf("Shor(15): p=%d, q=%d, attempts=%d\n", res.P, res.Q, res.Attempts)
}
