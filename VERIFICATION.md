# Verification — 2026-10-13

- Local lightweight suite: 10 passed; two torch-dependent test modules skipped
  because the local Python environment does not contain PyTorch.
- CPU suite with PyTorch and the pinned SPF scheduler: **20 passed**.
  One upstream NumPy deprecation warning, no test failures.
- Eleven original numerical functions have identical canonical AST signatures:
  flow prediction/loss/Euler sampling, extreme feature/loss/truncated-gradient
  functions, and rainfall-budget feature/loss/objective functions.
- Tests cover SSP126-only metadata and normalization rejection, full month
  coverage/no padded duplicates, all calendar lengths, finite auxiliary
  gradients, train-only checkpoint lineage, safe checkpoint deserialization,
  forbidden evaluation-GT conditions, and the original Euler endpoint.
- Only synthetic CPU tests ran. No climate data preparation, GPU smoke,
  DDP training, weather generation, official scoring or performance evaluation
  ran for this release. End-to-end four-GPU execution remains unverified.
- No old frozen source, checkpoint or dataset was modified. Runtime test
  dependency installation was isolated to a disposable test directory.
- The historical +0.12481 GT-calibrated result is NOT a release performance claim.
