#include "shor.h"
#include "parallel.h"
#include "qft.h"
#include "standart.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void apply_modular_exp(qreg *q,
                       int counting_start, int t,
                       int target_start,   int n,
                       uint64_t a, uint64_t N) {
    QREG_ASSERT(q != NULL, "apply_modular_exp: q is NULL");
    QREG_ASSERT(N >= 2,    "apply_modular_exp: N < 2");
    QREG_ASSERT(t >= 1 && n >= 1, "apply_modular_exp: t, n < 1");
    QREG_ASSERT(counting_start >= 0 && target_start >= 0,
                "apply_modular_exp: negative register start");
    QREG_ASSERT(counting_start + t <= q->n_qubits,
                "apply_modular_exp: counting range overflows");
    QREG_ASSERT(target_start + n <= q->n_qubits,
                "apply_modular_exp: target range overflows");
    QREG_ASSERT(counting_start + t <= target_start ||
                target_start   + n <= counting_start,
                "apply_modular_exp: counting and target ranges overlap");
    QREG_ASSERT(N <= ((uint64_t)1 << n),
                "apply_modular_exp: N > 2^target_width");
    QREG_ASSERT(gcd_u64(a, N) == 1,
                "apply_modular_exp: gcd(a, N) != 1");

    size_t base   = (size_t)q->rank * q->local_size;
    size_t t_mask = (((size_t)1 << t) - 1) << counting_start;
    size_t n_mask = (((size_t)1 << n) - 1) << target_start;
    size_t outer  = ~(t_mask | n_mask);

    /* Pre-count non-zero amplitudes to size the buffers. */
    size_t n_nz = 0;
    for (size_t i = 0; i < q->local_size; i++) {
        if (q->amp[i] != 0.0) n_nz++;
    }
    size_t         *idx = malloc(n_nz * sizeof *idx);
    complex double *val = malloc(n_nz * sizeof *val);
    size_t        cur = 0;
    for (size_t i = 0; i < q->local_size; i++) {
        if (q->amp[i] == 0.0) continue;
        size_t global = base + i;
        uint64_t x = (global >> counting_start) & (((uint64_t)1 << t) - 1);
        uint64_t y = (global >> target_start)   & (((uint64_t)1 << n) - 1);
        uint64_t y_new;
        if (y < N) y_new = (y * mod_pow(a, x, N)) % N;
        else       y_new = y;          /* reversibility pass-through */
        size_t new_global = (global & outer)
                          | ((size_t)x     << counting_start)
                          | ((size_t)y_new << target_start);
        idx[cur] = new_global;
        val[cur] = q->amp[i];
        cur++;
    }
    /* Zero the local slice; redistribute_pairs accumulates incoming. */
    for (size_t i = 0; i < q->local_size; i++) q->amp[i] = 0.0;
    redistribute_pairs(q, idx, n_nz, val);
    free(idx); free(val);
}

shor_period_result apply_shor_period(qreg *q,
                                     int counting_start, int t,
                                     int target_start,   int n,
                                     uint64_t a, uint64_t N) {
    QREG_ASSERT(q != NULL,                 "apply_shor_period: q is NULL");
    QREG_ASSERT(counting_start >= 0 && t >= 1, "bad counting range");
    QREG_ASSERT(target_start  >= 0 && n >= 1, "bad target range");
    /* Setup: counting register in uniform superposition, target in |1>. */
    /* Zero everything, then set the single basis state with y=1, x=0. */
    qreg_init_basis(q, (size_t)1 << target_start);
    /* Hadamards on the counting register. */
    for (int j = 0; j < t; j++) apply_h(q, counting_start + j);
    /* Apply modular exponentiation. */
    apply_modular_exp(q, counting_start, t, target_start, n, a, N);
    /* Inverse QFT on the counting register. */
    apply_qft_inverse(q, counting_start, t);
    /* Measure the counting register to get integer c. We measure each
     * qubit and combine; alternatively we could call measure_all and
     * mask out the counting bits. Use bitwise measure_qubit calls.       */
    uint64_t c = 0;
    for (int j = 0; j < t; j++) {
        int bit = measure_qubit(q, counting_start + j);
        c |= ((uint64_t)bit) << j;
    }
    /* Classical post-processing: continued-fraction expansion of c/2^t. */
    double x = (double)c / (double)((uint64_t)1 << t);
    uint64_t num = 0, den = 0;
    continued_fraction(x, N, &num, &den);
    shor_period_result res = { .r = den, .measured_c = c };
    return res;
}
static int s_seeded = 0;

shor_factor_result shor_factor(uint64_t N, int max_attempts) {
    QREG_ASSERT(N >= 4, "shor_factor: N too small");
    QREG_ASSERT(max_attempts > 0, "shor_factor: max_attempts < 1");
    shor_factor_result out = {0, 0, 0};
    /* Handle even N trivially. */
    if ((N & 1) == 0) { out.p = 2; out.q = N / 2; out.attempts = 0; return out; }
    /* Pick the bit width for the target register: n = ceil(log2 N).
     * Counting width: t = 2n + 1.                                       */
    int n = 0;
    while (((uint64_t)1 << n) < N) n++;
    int t = 2 * n + 1;
    int n_total = t + n;
    if (n_total > QREG_MAX_QUBITS) { out.attempts = 0; return out; }
    /* Seed once per process. Each rank uses its rank to differ. */
    int rank;  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (!s_seeded) { srand((unsigned)time(NULL) + (unsigned)rank); s_seeded = 1; }
    for (int attempt = 1; attempt <= max_attempts; attempt++) {
        out.attempts = attempt;
        uint64_t a = 2 + ((uint64_t)rand() % (N - 3));   /* in [2, N-2] */
        uint64_t g = gcd_u64(a, N);
        if (g > 1) { out.p = g; out.q = N / g; return out; }    /* lucky */
        qreg *q = qreg_create(n_total, MPI_COMM_WORLD);
        shor_period_result pr = apply_shor_period(q, /*cs=*/n, t, /*ts=*/0,
                                                  n, a, N);
        qreg_destroy(q);
        uint64_t r = pr.r;
        if (r == 0 || (r & 1)) continue;                 /* need even r */
        uint64_t x  = mod_pow(a, r / 2, N);
        if (x + 1 == N) continue;                        /* trivial */
        uint64_t p1 = gcd_u64(x + 1, N);
        uint64_t p2 = gcd_u64(x + N - 1, N);
        if (p1 > 1 && p1 < N) { out.p = p1; out.q = N / p1; return out; }
        if (p2 > 1 && p2 < N) { out.p = p2; out.q = N / p2; return out; }
    }
    return out;   /* p = q = 0 on failure */
}
