## 2026-11-06 - PyTorch Data Loading Optimization
**Learning:** The `neuralprognoz` codebase loads time-series and static data using PyTorch DataLoaders in `basetrainer.py` and `tester.py` but was not pinning memory or using asynchronous GPU transfers, representing a clear bottleneck in host-to-device memory bandwidth.
**Action:** Set `pin_memory=True` (when on GPU) and use `non_blocking=True` for `.to(device)` calls, alongside `zero_grad(set_to_none=True)` to maximize training and evaluation throughput.

## 2026-11-06 - Date parsing in custom loss functions
**Learning:** In PyTorch models that incorporate custom time-aware physical loss functions (like `SummerWeightedPhysicsLoss` and `JulyWeightedPhysicsLoss`), dynamically extracting the month from a batch of dates on each forward pass using `np.vectorize(lambda d: pd.to_datetime(d).month)` causes a massive CPU bottleneck because it iterates through the elements calling the Python lambda function rather than using true vectorization.
**Action:** Replace `np.vectorize` with `pd.DatetimeIndex(dates_np.ravel()).month.values.reshape(dates_np.shape)` which performs true vectorized datetime parsing and extraction, running approximately 600x faster and eliminating the CPU bottleneck in the loss calculation.
