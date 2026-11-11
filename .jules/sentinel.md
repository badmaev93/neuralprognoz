## 2024-11-06 - [CRITICAL] PyTorch Model Loading (torch.load) Insecure Deserialization
**Vulnerability:** `torch.load()` is used throughout the codebase (`tester.py`, `basetrainer.py`) to load model weights and optimizer states without specifying `weights_only=True`.
**Learning:** PyTorch's `torch.load()` uses Python's `pickle` module by default, which can execute arbitrary code during deserialization. Loading untrusted checkpoint files (`.pt` or `.pth`) is equivalent to remote code execution (RCE).
**Prevention:** Always use `weights_only=True` when calling `torch.load()` to load model weights (`state_dict`). This restricts the unpickler to only load tensors and primitive types, preventing arbitrary code execution.

## 2024-11-07 - [HIGH] Hardcoded Dummy API Key Fallback Risk
**Vulnerability:** A script (`timegpt_long_horizon.py`) was observed using a fallback dummy API key (`os.environ.get("TIMEGPT_API_KEY", "your_api_key_here")`). While not a real key, this practice is dangerous. It sets a pattern where developers might accidentally replace the placeholder with a real key during local testing and commit it. Additionally, it causes the application to proceed with a known bad state, leading to confusing downstream errors instead of failing fast securely.
**Learning:** Fallback values for sensitive credentials in configuration fetching can easily lead to accidental secret leaks or insecure failure modes.
**Prevention:** Never use dummy fallbacks for sensitive environment variables. Always fetch strictly (`os.environ.get("KEY")`) and raise an immediate, descriptive exception (e.g., `ValueError`) if the required secret is not provided in the environment. This ensures the application fails securely and fast.
