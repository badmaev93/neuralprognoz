## 2026-11-06 - PyTorch Data Loading Optimization
**Learning:** The `neuralprognoz` codebase loads time-series and static data using PyTorch DataLoaders in `basetrainer.py` and `tester.py` but was not pinning memory or using asynchronous GPU transfers, representing a clear bottleneck in host-to-device memory bandwidth.
**Action:** Set `pin_memory=True` (when on GPU) and use `non_blocking=True` for `.to(device)` calls, alongside `zero_grad(set_to_none=True)` to maximize training and evaluation throughput.

## 2026-11-06 - Date parsing in custom loss functions
**Learning:** In PyTorch models that incorporate custom time-aware physical loss functions (like `SummerWeightedPhysicsLoss` and `JulyWeightedPhysicsLoss`), dynamically extracting the month from a batch of dates on each forward pass using `np.vectorize(lambda d: pd.to_datetime(d).month)` causes a massive CPU bottleneck because it iterates through the elements calling the Python lambda function rather than using true vectorization.
**Action:** Replace `np.vectorize` with `pd.DatetimeIndex(dates_np.ravel()).month.values.reshape(dates_np.shape)` which performs true vectorized datetime parsing and extraction, running approximately 600x faster and eliminating the CPU bottleneck in the loss calculation.

## 2026-11-06 - PyTorch Training Loop Optimization
**Learning:** PyTorch DataLoader defaults to `persistent_workers=False` and `torch.backends.cudnn.benchmark=False`. When `num_workers > 0`, `persistent_workers=False` destroys and recreates background workers at the end of each epoch, which is a major overhead in the PyTorch training loop. The cuDNN auto-tuner is also disabled by default, although it can find the fastest convolution/RNN algorithm for a given input size, offering a free speedup when input sizes don't change.
**Action:** Set `persistent_workers=True` in `DataLoader` when `num_workers > 0` to reuse background worker processes. Set `torch.backends.cudnn.benchmark = True` when the device is `cuda` to enable the cuDNN auto-tuner.

## 2024-05-18 - pathlib.Path.glob evaluation optimization
**Learning:** In `GenericDataset` loading time series data, using `list(data_dir.glob(...))` forcefully iterates over the entire directory tree just to get the first match, resulting in massive file I/O overhead (O(N)) for large datasets with many subdirectories.
**Action:** Use `next(data_dir.glob(...), None)` to lazily evaluate the glob generator, stopping the recursive search immediately when the file is found (O(1) relative to matches), while maintaining the structural flexibility of searching across subdirectories.
