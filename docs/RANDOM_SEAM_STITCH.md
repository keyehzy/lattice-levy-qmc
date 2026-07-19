# Random-seam heat-bath Levy stitching

The global ideal-gas proposal is exact, but at finite interaction it redraws
the entire configuration. Its acceptance therefore falls rapidly with system
size. The stitch kernel changes a small set of `k` strands in a finite
imaginary-time slab while retaining exact free-particle conditionals. The
two-strand kernel remains the default.

## Fixed-seam proposal

Choose distinct labeled paths `p[0], ..., p[k-1]` and a slab
`[tau0, tau1]`. Their left endpoints are `a_i`, and their right endpoints are
`b_j`, all reduced on the torus. Define

```text
W[i,j] = K(a_i, b_j; T),
T = tau1 - tau0.
```

The code samples an endpoint permutation `sigma` with probability

```text
P(sigma | exterior) = product_i W[i,sigma(i)] / perm(W),
```

then independently samples the `k` exact torus bridges for that matching. It
joins prefix `i` to suffix `sigma(i)` and assigns
`new_successor[p[i]] = old_successor[p[sigma(i)]]`. Every proposed
configuration remains closed. For `k=2`, this is the original retain/exchange
heat bath that splits or merges cycles through a successor transposition.

The permanent and matching are evaluated in log space with the subset
recursion

```text
F(mask) = sum_{j not in mask} W[r,j] F(mask union {j}),
r = popcount(mask),
F(full_mask) = 1.
```

The implementation caps `k` at 8, making this `O(k 2^k)` recursion small.

Together, matching and bridge sampling are a draw from the exact ideal-gas
conditional distribution in the slab. In the Metropolis-Hastings ratio, its
free density cancels the free part of the target. The remaining acceptance is

```text
min(1, exp(-(S_U(new) - S_U(old)))).
```

Strand-selection probabilities may depend on the paths' left seam positions
because a stitch leaves those positions unchanged. Consequently the same
ordered selection probability appears in the forward and reverse moves. A
nonzero uniform-partner probability keeps the selection graph connected. A
fixed state-independent `StitchMixture`, such as counts `(2,3,4)` with weights
`(0.8,0.15,0.05)`, is therefore also reversible; counts larger than the fixed
canonical particle number are removed before normalizing the weights.

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
