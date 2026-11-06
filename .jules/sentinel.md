## 2024-11-06 - [CRITICAL] PyTorch Model Loading (torch.load) Insecure Deserialization
**Vulnerability:** `torch.load()` is used throughout the codebase (`tester.py`, `basetrainer.py`) to load model weights and optimizer states without specifying `weights_only=True`.
**Learning:** PyTorch's `torch.load()` uses Python's `pickle` module by default, which can execute arbitrary code during deserialization. Loading untrusted checkpoint files (`.pt` or `.pth`) is equivalent to remote code execution (RCE).
**Prevention:** Always use `weights_only=True` when calling `torch.load()` to load model weights (`state_dict`). This restricts the unpickler to only load tensors and primitive types, preventing arbitrary code execution.
