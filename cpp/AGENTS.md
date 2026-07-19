# C++ Codebase Guidelines

These guidelines apply to the C++ implementation under `cpp/`. The Python
reference implementation belongs under `python/`; keep language-specific
source, tests, build files, and generated artifacts inside their respective
trees. Shared design and mathematical documentation belongs in the repository
level `docs/` directory.

## Scope

- Support both the ideal retained-grid sampler and the exact continuous-time
  finite-`U` sampler. Keep their path representations distinct: the retained
  ideal grid is not sufficient for the interacting action.
- Treat the corresponding Python implementations and tests as behavioral
  references. Preserve probability laws, numerical tolerances, coordinate
  conventions, event-boundary semantics, winding sectors, permutation
  endpoints, and public results; byte-for-byte random streams are not required.
- Finite-`U` proposals must be exact free conditional or independence draws so
  their Metropolis ratio contains only the interaction-action difference.
- Consult `docs/ARCHITECTURE.md` for algorithm semantics and
  `docs/CPP_PORT.md` for the porting contract and suggested interfaces.

## Project layout

Use a conventional library layout:

```text
cpp/
  CMakeLists.txt
  include/qmc/       public headers
  src/               implementation files and private headers
  tests/             C++ tests and statistical regressions
  examples/          small executable examples
```

Keep the core sampler usable as a library. Command-line programs and examples
must call library APIs rather than contain sampling logic. Avoid dependencies
from C++ code back into Python; cross-language comparison scripts may live in
the Python tree.

## Language and build

- Target C++20 and use standard-library facilities where they are sufficient.
- Build with CMake through target-based configuration. Do not set global
  compiler flags or global include directories.
- Keep the default build warning-clean with Clang and GCC. CI and local debug
  builds should enable strict warnings; warnings must not be suppressed without
  a documented reason.
- Make optional tools, examples, sanitizers, and tests explicit CMake options.
- Keep third-party dependencies few and justified. Prefer small, focused
  libraries with CMake package support, and do not vendor generated dependency
  trees into the repository.
- Never commit build directories, binaries, coverage output, or generated
  dependency files.

## Design and APIs

- Put production code in the `qmc` namespace. Headers under `include/qmc/`
  define the supported public surface; implementation details remain private.
- Separate immutable model/numerical parameters from sampled configuration
  state. Make ownership explicit and prefer value semantics and RAII.
- Pass the random-number generator explicitly to every stochastic operation.
  Do not use global or hidden thread-local RNG state.
- Use fixed-width integer types where representation matters. Covering-space
  coordinates must be signed and wide enough for winding displacements.
- Use `std::span` for non-owning contiguous inputs and `std::vector` for
  variable-sized owned data unless measurements justify another container.
- Express invariants in types and constructors where practical. Validate all
  public inputs at API boundaries and report invalid parameters with clear
  exceptions; use assertions only for internal programmer errors.
- Mark important return values `[[nodiscard]]`, use `const` consistently, and
  apply `noexcept` only when the guarantee is real and useful.
- Avoid premature class hierarchies, generic frameworks, and mutable singletons.
  Prefer small functions and composition.

## Numerical and sampling rules

- Use `double` for probabilities, times, log weights, and numerical work unless
  a test demonstrates the need for another representation.
- Evaluate partition recursions and probability normalizations in log space.
  Centralize stable primitives such as log-sum-exp and discrete CDF inversion.
- Preserve covering-space paths separately from coordinates reduced modulo
  `L`; winding information must never be inferred from reduced coordinates.
- Preserve the meaning of `M`: it controls retained bridge points in the ideal
  sampler and is not a Trotter approximation. The interacting sampler is sparse
  and event-driven and has no `M` parameter.
- Make truncation controls such as tail tolerance and maximum support explicit
  numerical options. A truncated discrete distribution must have a documented
  bound on omitted probability mass and fail clearly if its safety limit is
  reached.
- Check integer arithmetic used for volumes, sizes, indices, displacements,
  and allocation counts before it can overflow.
- Never change a probability law for speed without documenting the derivation
  and adding distribution-level regression tests.

## Style

- Follow the existing `.clang-format` and `.clang-tidy` configurations once
  present. Until then, use a consistent LLVM-like style with 2-space
  indentation and a 100-column target.
- Use `snake_case` for functions and variables, `PascalCase` for types, and
  `kPascalCase` for named constants. Name files in `snake_case`.
- Include what each file uses. Order includes as the matching header, C++
  standard library, third-party libraries, then project headers.
- Keep headers self-contained and minimize transitive includes. Do not place
  `using namespace` directives in headers.
- Comments should explain mathematical intent, invariants, numerical bounds,
  and non-obvious decisions. Do not narrate straightforward code.
- Public APIs and probability kernels need concise documentation of units,
  domains, endpoint conventions, ownership, and failure behavior.

## Tests and verification

- Every production change must add or update focused C++ tests. A bug fix must
  include a regression test that fails without the fix.
- Divide tests into deterministic unit tests, structural/property tests, and
  statistical tests. Keep slow statistical tests labeled separately from the
  fast default suite.
- Test exact invariants: canonical recursion base cases, valid permutations and
  cycle partitions, endpoint stitching, modulo reduction, covering
  displacement, winding divisibility, dimensions, and finite outputs.
- Compare deterministic numerical tables and seeded aggregate statistics with
  the Python reference on small systems. Use tolerances justified by floating-
  point error or sampling uncertainty, not tolerances chosen merely to make a
  test pass.
- Do not require individual C++ samples to match NumPy samples. Reproducibility
  means the same C++ build and seed produce the same result under the supported
  RNG contract.
- Run unit tests and relevant sanitizers before merging. Benchmark before and
  after performance-oriented changes, and record the workload and compiler.

## Change discipline

- Keep commits and reviews narrow: separate mechanical reorganization,
  algorithm ports, refactors, and optimizations when possible.
- Port behavior first, verify it against Python, and optimize only measured
  bottlenecks without weakening invariants or tests.
- Update repository documentation when a public API, algorithm, layout, build
  requirement, or supported platform changes.
- Do not duplicate the mathematical specification in C++ comments. Link to the
  relevant repository documentation and document only implementation-specific
  details nearby.
