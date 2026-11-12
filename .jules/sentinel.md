## 2024-11-06 - [CRITICAL] PyTorch Model Loading (torch.load) Insecure Deserialization
**Vulnerability:** `torch.load()` is used throughout the codebase (`tester.py`, `basetrainer.py`) to load model weights and optimizer states without specifying `weights_only=True`.
**Learning:** PyTorch's `torch.load()` uses Python's `pickle` module by default, which can execute arbitrary code during deserialization. Loading untrusted checkpoint files (`.pt` or `.pth`) is equivalent to remote code execution (RCE).
**Prevention:** Always use `weights_only=True` when calling `torch.load()` to load model weights (`state_dict`). This restricts the unpickler to only load tensors and primitive types, preventing arbitrary code execution.

## 2024-11-08 - [HIGH] Insecure Deserialization via pickle.load
**Vulnerability:** `pickle.load()` is widely used to deserialize untrusted dataset and result files (e.g., in `basedataset.py` and `np_run.py`).
**Learning:** `pickle` is inherently unsafe as it allows arbitrary Python code execution upon unpickling. Migrating away from `pickle` in data manipulation libraries (like pandas and xarray) is difficult, but user awareness is critical.
**Prevention:** Added warning log messages immediately before calling `pickle.load()` to inform users to only load files from trusted sources. Wherever possible, `numpy.load(allow_pickle=False)` or safer formats like JSON/HDF5 should be used.

## 2024-11-12 - [CRITICAL] Hardcoded API Key Fallbacks
**Vulnerability:** `os.environ.get("TIMEGPT_API_KEY", "your_api_key_here")` used a dummy string as a fallback for a missing API key.
**Learning:** Hardcoded fallbacks for credentials or API keys can mask configuration errors, lead to confusing authentication failures downstream, and violate secure-by-default principles. Applications should fail fast when critical environment variables are absent.
**Prevention:** Strictly validate the presence of sensitive environment variables and raise an immediate `ValueError` if they are missing, rather than supplying a fallback.
