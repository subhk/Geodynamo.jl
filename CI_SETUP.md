Continuous Integration (CI) notes for Geodynamo.jl with SHTnsKit

Summary
- Tests require SHTnsKit.jl available locally (path source).
- Basic, single-rank smoke tests can run without mpirun; multi-rank testing needs MPI runner.

GitHub Actions (suggested)
1) Check out this repo and SHTnsKit.jl side-by-side, so the path source `../SHTnsKit.jl` resolves:

  - uses: actions/checkout@v4
  - name: Checkout SHTnsKit
    uses: actions/checkout@v4
    with:
      repository: <your-org-or-user>/SHTnsKit.jl
      path: SHTnsKit.jl
  - name: Create path alias
    run: ln -s "$GITHUB_WORKSPACE/SHTnsKit.jl" "$GITHUB_WORKSPACE/../SHTnsKit.jl"

2) Set up Julia and run tests:

  - uses: julia-actions/setup-julia@v1
    with:
      version: '1.10'
  - name: Instantiate
    run: julia --project=. -e 'using Pkg; Pkg.instantiate()'
  - name: Run tests (single rank)
    run: julia --project=. -e 'using Pkg; Pkg.test(; test_args=["--quick"])'

MPI (optional)
- To run multi-rank tests, install MPI runtime in CI and call:
  mpirun -n 2 julia --project=. extras/smoke_test.jl

Local testing
- Single process: julia --project=. extras/smoke_test.jl
- Multi-process:  mpirun -n 4 julia --project=. extras/smoke_test.jl

Wiring tests
- The repository contains an extras test script: extras/test_shtnskit_roundtrip.jl
- To run it under Pkg.test, add this line to test/runtests.jl:
  include("../extras/test_shtnskit_roundtrip.jl")

