# Random-seam heat-bath Levy stitching

The global ideal-gas proposal is exact, but at finite interaction it redraws
the entire configuration. Its acceptance therefore falls rapidly with system
size. The stitch kernel changes only two strands in a finite imaginary-time
slab while retaining exact free-particle conditionals.

## Fixed-seam proposal

Choose two distinct labeled paths `i` and `j` and a slab
`[tau0, tau1]`. Their left endpoints are `a_i, a_j`, and their right
endpoints are `b_i, b_j`, all reduced on the torus. There are two possible
matchings across the slab. Their free weights are

```text
w_identity = K(a_i, b_i; T) K(a_j, b_j; T)
w_exchange = K(a_i, b_j; T) K(a_j, b_i; T),
T = tau1 - tau0.
```

The code samples the matching with probability proportional to these weights,
then independently samples the two exact torus bridges for that matching. An
exchange swaps the two right suffixes and transposes the successors of `i` and
`j`; this splits one permutation cycle or merges two cycles. Every proposed
configuration remains closed.

Together, matching and bridge sampling are a draw from the exact ideal-gas
conditional distribution in the slab. In the Metropolis-Hastings ratio, its
free density cancels the free part of the target. The remaining acceptance is

```text
min(1, exp(-(S_U(new) - S_U(old)))).
```

Partner probabilities may depend on the paths' left seam positions because a
stitch leaves those positions unchanged. Consequently the same ordered-pair
selection probability appears in the forward and reverse moves. A nonzero
uniform-partner probability keeps the pair-selection graph connected.

## Random seam and reversibility

Building spatial partner buckets costs `O(N)`, so a sweep performs several
attempts at one fixed seam. To avoid favoring that seam, the macro-kernel is

```text
A B^m A,
```

where `A` is a uniform cyclic rotation of all closed loops in imaginary time
and `B` is one fixed-seam random-scan stitch update. Uniform rotation is a
self-adjoint projection, and `B` is self-adjoint. Therefore `A B^m A` is also
self-adjoint and satisfies detailed balance.

## Local action calculation

The accepted configuration owns a sparse per-site occupancy timeline. Before
replacing paths, their overlaps with the current occupancy are removed one at
a time; proposed paths are then added one at a time. This counts every affected
pair exactly once. A rejection reverses the ledger edits. Tests periodically
compare the cached result with a full event sweep.

The locality changes proposal cost and acceptance, not the target measure.
For small systems, the original global move remains useful as an independent
cross-check or as part of a hybrid schedule.
