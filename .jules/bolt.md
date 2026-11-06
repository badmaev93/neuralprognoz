## 2026-11-06 - PyTorch Data Loading Optimization
**Learning:** The `neuralprognoz` codebase loads time-series and static data using PyTorch DataLoaders in `basetrainer.py` and `tester.py` but was not pinning memory or using asynchronous GPU transfers, representing a clear bottleneck in host-to-device memory bandwidth.
**Action:** Set `pin_memory=True` (when on GPU) and use `non_blocking=True` for `.to(device)` calls, alongside `zero_grad(set_to_none=True)` to maximize training and evaluation throughput.
