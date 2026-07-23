# Vendored dependencies

## TRIQS/maxent

`triqs_maxent/` is an unmodified source snapshot of
[TRIQS/maxent](https://github.com/TRIQS/maxent):

- upstream version: `4.0.0`
- upstream commit: `79926af3a6e49310570839d19cf3bc1a917778c2`
- snapshot date: 2026-07-23
- license: GNU GPL version 3 or later; see
  `triqs_maxent/COPYING.txt`, `triqs_maxent/LICENSE`, and
  `triqs_maxent/LICENSE.txt`

The repository-level adapter in `python/qmc_maxent.py` uses the vendored
pure-Python `DataKernel` and `MaxEntLoop` implementation. It deliberately does
not modify the upstream source and does not require a system TRIQS installation
for this array-based integration. The rest of the upstream package retains its
normal TRIQS 4.0 build requirement.

To refresh the snapshot, export a tagged upstream tree rather than committing
its nested `.git` directory, then update the version and commit recorded above
and in `python/qmc_maxent.py`.
