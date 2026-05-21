#include "matrix.h"
#include "parallel.h"
#include "standart.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

qreg *qreg_create(int n_qubits, MPI_Comm comm) {
    if (n_qubits < 1 || n_qubits > QREG_MAX_QUBITS) return NULL;

    int n_procs, rank;
    MPI_Comm_size(comm, &n_procs);
    MPI_Comm_rank(comm, &rank);
    if (!is_power_of_two(n_procs))       return NULL;
    if ((size_t)n_procs > ((size_t)1 << n_qubits)) return NULL;

    qreg *q = (qreg *)malloc(sizeof *q);
    if (!q) return NULL;
    q->n_qubits   = n_qubits;
    q->n_procs    = n_procs;
    q->rank       = rank;
    q->p          = ilog2_u32((uint32_t)n_procs);
    q->local_size = (size_t)1 << (n_qubits - q->p);
    q->comm       = comm;
    q->amp        = (complex double *)calloc(q->local_size, sizeof *q->amp);
    if (!q->amp) { free(q); return NULL; }
    return q;
}

void qreg_destroy(qreg *q) {
    if (!q) return;
    free(q->amp);
    free(q);
}

void qreg_init_basis(qreg *q, size_t basis_state) {
    QREG_ASSERT(q != NULL, "qreg_init_basis: q is NULL");
    QREG_ASSERT(basis_state < ((size_t)1 << q->n_qubits),
                "qreg_init_basis: basis_state out of range");
    /* zero everything */
    for (size_t i = 0; i < q->local_size; i++) q->amp[i] = 0.0;
    /* set the owning rank's amplitude to 1 */
    int owning_rank = (int)(basis_state >> (q->n_qubits - q->p));
    if (q->rank == owning_rank) {
        size_t off = basis_state & (q->local_size - 1);
        q->amp[off] = 1.0;
    }
}

double qreg_norm(const qreg *q) {
    QREG_ASSERT(q != NULL, "qreg_norm: q is NULL");
    double local = 0.0;
    for (size_t i = 0; i < q->local_size; i++) {
        double r = creal(q->amp[i]);
        double im = cimag(q->amp[i]);
        local += r*r + im*im;
    }
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, q->comm);
    return global;
}

double prob_of(const qreg *q, size_t basis) {
    QREG_ASSERT(q != NULL, "prob_of: q is NULL");
    QREG_ASSERT(basis < ((size_t)1 << q->n_qubits),
                "prob_of: basis out of range");
    int owning_rank = (int)(basis >> (q->n_qubits - q->p));
    double local = 0.0;
    if (q->rank == owning_rank) {
        size_t off = basis & (q->local_size - 1);
        double r  = creal(q->amp[off]);
        double im = cimag(q->amp[off]);
        local = r*r + im*im;
    }
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, q->comm);
    return global;
}

static void apply_u_local(qreg *q, int target, complex double u[2][2]) {
    size_t stride = (size_t)1 << target;
    size_t step   = stride << 1;
    for (size_t base = 0; base < q->local_size; base += step) {
        for (size_t off = 0; off < stride; off++) {
            size_t i0 = base + off;
            size_t i1 = i0 + stride;
            complex double a0 = q->amp[i0];
            complex double a1 = q->amp[i1];
            q->amp[i0] = u[0][0]*a0 + u[0][1]*a1;
            q->amp[i1] = u[1][0]*a0 + u[1][1]*a1;
        }
    }
}

static void apply_u_global(qreg *q, int target, complex double u[2][2]) {
    int   tbit    = target - (q->n_qubits - q->p);
    int   mybit   = (q->rank >> tbit) & 1;
    int   partner = q->rank ^ (1 << tbit);
    complex double *buf = malloc(q->local_size * sizeof *buf);
    exchange_amplitudes(q, partner, buf);
    /* Combine our slice with the partner's slice. We hold the value of
     * qubit `target` == mybit; partner held the opposite bit. The pair
     * (a_mybit, a_{1-mybit}) corresponds to amplitude (q->amp[i], buf[i]).
     */
    if (mybit == 0) {
        for (size_t i = 0; i < q->local_size; i++) {
            complex double a0 = q->amp[i];
            complex double a1 = buf[i];
            q->amp[i] = u[0][0]*a0 + u[0][1]*a1;
        }
    } else {
        for (size_t i = 0; i < q->local_size; i++) {
            complex double a0 = buf[i];
            complex double a1 = q->amp[i];
            q->amp[i] = u[1][0]*a0 + u[1][1]*a1;
        }
    }
    free(buf);
}

void apply_u(qreg *q, int target, complex double u[2][2]) {
    QREG_ASSERT(q != NULL, "apply_u: q is NULL");
    QREG_ASSERT(u != NULL, "apply_u: u is NULL");
    QREG_ASSERT(target >= 0 && target < q->n_qubits,
                "apply_u: target out of range");
    if (is_local_qubit(q, target)) apply_u_local (q, target, u);
    else                           apply_u_global(q, target, u);
}

void apply_h(qreg *q, int target) {
    const double s = 1.0 / sqrt(2.0);
    complex double u[2][2] = {
        { s,  s},
        { s, -s},
    };
    apply_u(q, target, u);
}

void apply_x(qreg *q, int target) {
    complex double u[2][2] = { {0, 1}, {1, 0} };
    apply_u(q, target, u);
}
void apply_y(qreg *q, int target) {
    complex double u[2][2] = { {0, -I}, {I, 0} };
    apply_u(q, target, u);
}
void apply_z(qreg *q, int target) {
    complex double u[2][2] = { {1, 0}, {0, -1} };
    apply_u(q, target, u);
}

void apply_phase(qreg *q, int target, double theta) {
    complex double u[2][2] = { {1, 0}, {0, cexp(I * theta)} };
    apply_u(q, target, u);
}
void apply_s(qreg *q, int target) { apply_phase(q, target, M_PI / 2.0); }
void apply_t(qreg *q, int target) { apply_phase(q, target, M_PI / 4.0); }

void apply_rx(qreg *q, int target, double theta) {
    double c = cos(theta / 2.0);
    double s = sin(theta / 2.0);
    complex double u[2][2] = {
        { c,        -I * s },
        { -I * s,    c     },
    };
    apply_u(q, target, u);
}
void apply_ry(qreg *q, int target, double theta) {
    double c = cos(theta / 2.0);
    double s = sin(theta / 2.0);
    complex double u[2][2] = {
        { c, -s },
        { s,  c },
    };
    apply_u(q, target, u);
}
void apply_rz(qreg *q, int target, double theta) {
    complex double e_minus = cexp(-I * theta / 2.0);
    complex double e_plus  = cexp( I * theta / 2.0);
    complex double u[2][2] = {
        { e_minus, 0       },
        { 0,       e_plus  },
    };
    apply_u(q, target, u);
}

static void apply_cu_both_local(qreg *q, int control, int target,
                                complex double u[2][2]) {
    size_t cmask = (size_t)1 << control;
    size_t tstride = (size_t)1 << target;
    for (size_t i = 0; i < q->local_size; i++) {
        if ((i & cmask) && !(i & tstride)) {
            size_t j = i | tstride;
            complex double a0 = q->amp[i];
            complex double a1 = q->amp[j];
            q->amp[i] = u[0][0]*a0 + u[0][1]*a1;
            q->amp[j] = u[1][0]*a0 + u[1][1]*a1;
        }
    }
}

static void apply_cu_c_local_t_global(qreg *q, int control, int target,
                                      complex double u[2][2]) {
    int tbit    = target - (q->n_qubits - q->p);
    int mybit   = (q->rank >> tbit) & 1;
    int partner = q->rank ^ (1 << tbit);
    size_t cmask = (size_t)1 << control;
    complex double *buf = malloc(q->local_size * sizeof *buf);
    exchange_amplitudes(q, partner, buf);
    if (mybit == 0) {
        for (size_t i = 0; i < q->local_size; i++) {
            if (!(i & cmask)) continue;
            complex double a0 = q->amp[i];
            complex double a1 = buf[i];
            q->amp[i] = u[0][0]*a0 + u[0][1]*a1;
        }
    } else {
        for (size_t i = 0; i < q->local_size; i++) {
            if (!(i & cmask)) continue;
            complex double a0 = buf[i];
            complex double a1 = q->amp[i];
            q->amp[i] = u[1][0]*a0 + u[1][1]*a1;
        }
    }
    free(buf);
}

static void apply_cu_c_global_t_local(qreg *q, int control, int target,
                                      complex double u[2][2]) {
    int cbit = control - (q->n_qubits - q->p);
    if (((q->rank >> cbit) & 1) == 0) return;   /* no-op for this rank   */
    /* Otherwise the gate is just a local single-qubit u on the target.  */
    size_t tstride = (size_t)1 << target;
    size_t step    = tstride << 1;
    for (size_t base = 0; base < q->local_size; base += step) {
        for (size_t off = 0; off < tstride; off++) {
            size_t i0 = base + off;
            size_t i1 = i0 + tstride;
            complex double a0 = q->amp[i0];
            complex double a1 = q->amp[i1];
            q->amp[i0] = u[0][0]*a0 + u[0][1]*a1;
            q->amp[i1] = u[1][0]*a0 + u[1][1]*a1;
        }
    }
}

static void apply_cu_both_global(qreg *q, int control, int target,
                                 complex double u[2][2]) {
    /* Partner is by target bit; control bit is fixed per rank.         */
    int tbit    = target  - (q->n_qubits - q->p);
    int cbit    = control - (q->n_qubits - q->p);
    if (((q->rank >> cbit) & 1) == 0) return;     /* no-op on this rank */
    int mybit   = (q->rank >> tbit) & 1;
    int partner = q->rank ^ (1 << tbit);
    complex double *buf = malloc(q->local_size * sizeof *buf);
    exchange_amplitudes(q, partner, buf);
    if (mybit == 0) {
        for (size_t i = 0; i < q->local_size; i++) {
            complex double a0 = q->amp[i];
            complex double a1 = buf[i];
            q->amp[i] = u[0][0]*a0 + u[0][1]*a1;
        }
    } else {
        for (size_t i = 0; i < q->local_size; i++) {
            complex double a0 = buf[i];
            complex double a1 = q->amp[i];
            q->amp[i] = u[1][0]*a0 + u[1][1]*a1;
        }
    }
    free(buf);
}

void apply_cu(qreg *q, int control, int target, complex double u[2][2]) {
    QREG_ASSERT(q != NULL,           "apply_cu: q is NULL");
    QREG_ASSERT(u != NULL,           "apply_cu: u is NULL");
    QREG_ASSERT(control >= 0 && control < q->n_qubits,
                "apply_cu: control out of range");
    QREG_ASSERT(target  >= 0 && target  < q->n_qubits,
                "apply_cu: target out of range");
    QREG_ASSERT(control != target,
                "apply_cu: control == target");
    int c_local = is_local_qubit(q, control);
    int t_local = is_local_qubit(q, target);
    if      ( c_local &&  t_local) apply_cu_both_local      (q, control, target, u);
    else if ( c_local && !t_local) apply_cu_c_local_t_global(q, control, target, u);
    else if (!c_local &&  t_local) apply_cu_c_global_t_local(q, control, target, u);
    else                           apply_cu_both_global     (q, control, target, u);
}

void apply_cnot(qreg *q, int control, int target) {
    complex double u[2][2] = { {0, 1}, {1, 0} };
    apply_cu(q, control, target, u);
}
void apply_cz(qreg *q, int control, int target) {
    complex double u[2][2] = { {1, 0}, {0, -1} };
    apply_cu(q, control, target, u);
}
void apply_controlled_phase(qreg *q, int control, int target, double theta) {
    complex double u[2][2] = { {1, 0}, {0, cexp(I * theta)} };
    apply_cu(q, control, target, u);
}

void apply_swap(qreg *q, int a, int b) {
    QREG_ASSERT(q != NULL, "apply_swap: q is NULL");
    QREG_ASSERT(a != b, "apply_swap: a == b");
    apply_cnot(q, a, b);
    apply_cnot(q, b, a);
    apply_cnot(q, a, b);
}

void apply_multi_controlled_z(qreg *q, int n) {
    QREG_ASSERT(q != NULL,                 "apply_multi_controlled_z: q is NULL");
    QREG_ASSERT(n >= 1 && n <= q->n_qubits,"apply_multi_controlled_z: n out of range");
    /* Target the single basis state |1...1> on the first n qubits, with
     * higher bits (n .. n_qubits-1) free. We must phase-flip every
     * amplitude whose lower n bits are all 1 -- i.e. amp index with
     * (mask & i) == mask where mask = (1<<n)-1.
     * Iterate locally over indices that meet the predicate.              */
    size_t mask = ((size_t)1 << n) - 1;
    size_t base = (size_t)q->rank * q->local_size;
    for (size_t off = 0; off < q->local_size; off++) {
        size_t global = base + off;
        if ((global & mask) == mask) q->amp[off] = -q->amp[off];
    }
}

void apply_multi_controlled_x(qreg *q, const int *controls, int n_controls,
                              int target) {
    QREG_ASSERT(q != NULL,        "apply_mcx: q is NULL");
    QREG_ASSERT(controls != NULL, "apply_mcx: controls is NULL");
    QREG_ASSERT(n_controls >= 1,  "apply_mcx: at least one control required");
    QREG_ASSERT(target >= 0 && target < q->n_qubits,
                "apply_mcx: target out of range");
    /* Validate controls in range and distinct from target. */
    for (int i = 0; i < n_controls; i++) {
        QREG_ASSERT(controls[i] >= 0 && controls[i] < q->n_qubits,
                    "apply_mcx: control out of range");
        QREG_ASSERT(controls[i] != target,
                    "apply_mcx: control equals target");
        for (int j = i + 1; j < n_controls; j++)
            QREG_ASSERT(controls[i] != controls[j],
                        "apply_mcx: duplicate control");
    }
    /* V1 supports the case where every control AND the target is local.
     * The distributed multi-controlled X would itself decompose into
     * Toffoli + ancilla ladder; left for a follow-up.                    */
    for (int i = 0; i < n_controls; i++)
        QREG_ASSERT(is_local_qubit(q, controls[i]),
                    "apply_mcx: distributed controls not yet supported");
    QREG_ASSERT(is_local_qubit(q, target),
                "apply_mcx: distributed target not yet supported");
    size_t cmask = 0;
    for (int i = 0; i < n_controls; i++) cmask |= ((size_t)1 << controls[i]);
    size_t tstride = (size_t)1 << target;
    for (size_t i = 0; i < q->local_size; i++) {
        if ((i & cmask) == cmask && !(i & tstride)) {
            size_t j = i | tstride;
            complex double t = q->amp[i];
            q->amp[i] = q->amp[j];
            q->amp[j] = t;
        }
    }
}

int measure_qubit(qreg *q, int target) {
    QREG_ASSERT(q != NULL, "measure_qubit: q is NULL");
    QREG_ASSERT(target >= 0 && target < q->n_qubits,
                "measure_qubit: target out of range");
    /* 1. Compute P(bit_target == 0) locally; for each amplitude check
     *    whether its global index has bit_target = 0.                    */
    size_t base = (size_t)q->rank * q->local_size;
    double local_p0 = 0.0;
    for (size_t i = 0; i < q->local_size; i++) {
        size_t global = base + i;
        if (((global >> target) & 1) == 0) {
            double r = creal(q->amp[i]), im = cimag(q->amp[i]);
            local_p0 += r*r + im*im;
        }
    }
    double p0 = 0.0;
    MPI_Allreduce(&local_p0, &p0, 1, MPI_DOUBLE, MPI_SUM, q->comm);
    /* 2. Rank 0 samples; broadcast the outcome. */
    int outcome = 0;
    if (q->rank == 0) {
        double u = (double)rand() / (double)RAND_MAX;
        outcome = (u < p0) ? 0 : 1;
    }
    MPI_Bcast(&outcome, 1, MPI_INT, 0, q->comm);
    /* 3. Project and renormalise. */
    double p_observed = (outcome == 0) ? p0 : (1.0 - p0);
    double inv_sqrt   = 1.0 / sqrt(p_observed);
    for (size_t i = 0; i < q->local_size; i++) {
        size_t global = base + i;
        int bit = (int)((global >> target) & 1);
        if (bit != outcome) q->amp[i] = 0.0;
        else                q->amp[i] *= inv_sqrt;
    }
    return outcome;
}

size_t measure_all(qreg *q) {
    QREG_ASSERT(q != NULL, "measure_all: q is NULL");
    /* Build cumulative probability ranges per rank. Each rank computes
     * its local total |a_i|^2, then we do an MPI_Exscan to get the
     * running offset, then rank 0 samples u in [0,1) and broadcasts.
     * Each rank checks whether u falls in its range; the owning rank
     * picks the basis state with index found via linear scan.            */
    double local_total = 0.0;
    for (size_t i = 0; i < q->local_size; i++) {
        double r = creal(q->amp[i]), im = cimag(q->amp[i]);
        local_total += r*r + im*im;
    }
    double prefix = 0.0;
    MPI_Exscan(&local_total, &prefix, 1, MPI_DOUBLE, MPI_SUM, q->comm);
    if (q->rank == 0) prefix = 0.0;
    double u = 0.0;
    if (q->rank == 0) u = (double)rand() / (double)RAND_MAX;
    MPI_Bcast(&u, 1, MPI_DOUBLE, 0, q->comm);

    size_t chosen_global = 0;
    int    chosen_rank   = -1;
    if (u >= prefix && u < prefix + local_total) {
        double cum = prefix;
        for (size_t i = 0; i < q->local_size; i++) {
            double r = creal(q->amp[i]), im = cimag(q->amp[i]);
            cum += r*r + im*im;
            if (cum >= u) {
                chosen_global = (size_t)q->rank * q->local_size + i;
                chosen_rank = q->rank;
                break;
            }
        }
    }
    /* Allreduce to find the chosen rank/global index. */
    int picks_in = (chosen_rank == q->rank) ? q->rank : -1;
    int max_rank = -1;
    MPI_Allreduce(&picks_in, &max_rank, 1, MPI_INT, MPI_MAX, q->comm);
    size_t global_out = chosen_global;
    MPI_Bcast(&global_out, 1, MPI_UNSIGNED_LONG, max_rank, q->comm);
    /* Collapse the state: every amplitude except global_out is zeroed. */
    for (size_t i = 0; i < q->local_size; i++) q->amp[i] = 0.0;
    if (rank_owns(q, global_out)) {
        q->amp[global_to_local(q, global_out)] = 1.0;
    }
    return global_out;
}

void sample_distribution(const qreg *q, size_t *out, int shots) {
    QREG_ASSERT(q != NULL && out != NULL, "sample_distribution: NULL arg");
    QREG_ASSERT(shots > 0,                 "sample_distribution: shots <= 0");
    /* Naive: clone state, measure_all, restore. Inefficient but correct.
     * V1 is single-shot quality; a future version could compute the CDF
     * once and sample.                                                    */
    qreg *temp = qreg_clone(q);
    for (int s = 0; s < shots; s++) {
        /* Reset the clone to the original each shot. */
        memcpy(temp->amp, q->amp, q->local_size * sizeof *q->amp);
        out[s] = measure_all(temp);
    }
    qreg_destroy(temp);
}

qreg *qreg_clone(const qreg *q) {
    QREG_ASSERT(q != NULL, "qreg_clone: q is NULL");
    qreg *c = qreg_create(q->n_qubits, q->comm);
    if (!c) return NULL;
    memcpy(c->amp, q->amp, q->local_size * sizeof *q->amp);
    return c;
}

void qreg_dump(const qreg *q, FILE *f) {
    QREG_ASSERT(q != NULL && f != NULL, "qreg_dump: NULL arg");
    /* Gather to rank 0 and print. */
    size_t total = (size_t)1 << q->n_qubits;
    complex double *full = NULL;
    if (q->rank == 0) full = malloc(total * sizeof *full);
    MPI_Gather(q->amp,                       (int)q->local_size, MPI_C_DOUBLE_COMPLEX,
               full,                          (int)q->local_size, MPI_C_DOUBLE_COMPLEX,
               0, q->comm);
    if (q->rank == 0) {
        fprintf(f, "qreg(%d qubits, %zu amplitudes):\n", q->n_qubits, total);
        for (size_t i = 0; i < total; i++) {
            if (cabs(full[i]) < 1e-12) continue;
            fprintf(f, "  |%zu> = (%+.6f, %+.6f)\n",
                    i, creal(full[i]), cimag(full[i]));
        }
        free(full);
    }
}
