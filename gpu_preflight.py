"""
GPU Preflight: Ensure PyTorch has CUDA support when an NVIDIA GPU is present.

This script runs before the main backend starts. It performs these checks:
- Detects NVIDIA GPU using `nvidia-smi` (if available)
- Reports Python and PyTorch environment
- If a GPU is present but PyTorch lacks CUDA, attempts to install the correct CUDA wheels
  using PyTorch's official wheel index:
    1) Try cu121 (CUDA 12.1)
    2) Fallback to cu118 (CUDA 11.8)

Notes:
- The wheels include the CUDA runtime; you only need a sufficiently new NVIDIA driver.
- No changes are made if no NVIDIA GPU is detected.
"""

import os
import re
import sys
import json
import shutil
import subprocess
from typing import Optional, Tuple


def run(cmd: list[str], timeout: Optional[int] = 60) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            shell=False,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as e:
        return 1, "", str(e)


def detect_nvidia_gpu() -> Tuple[bool, dict]:
    info = {"has_nvidia": False, "driver_version": None, "cuda_version": None}
    # Prefer nvidia-smi if available in PATH
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return False, info

    code, out, err = run([nvsmi])
    if code != 0:
        return False, info

    info["has_nvidia"] = True
    # Parse driver and CUDA versions from nvidia-smi header
    # Example header: "Driver Version: 551.61    CUDA Version: 12.4"
    driver_match = re.search(r"Driver Version:\s*([0-9.]+)", out)
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", out)
    if driver_match:
        info["driver_version"] = driver_match.group(1)
    if cuda_match:
        info["cuda_version"] = cuda_match.group(1)
    return True, info


def torch_env() -> dict:
    env = {
        "torch_imported": False,
        "torch_version": None,
        "torch_cuda_version": None,
        "cuda_is_available": False,
        "error": None,
    }
    try:
        import torch  # type: ignore

        env["torch_imported"] = True
        env["torch_version"] = getattr(torch, "__version__", None)
        env["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
        try:
            env["cuda_is_available"] = bool(torch.cuda.is_available())
        except Exception:
            env["cuda_is_available"] = False
    except Exception as e:
        env["error"] = str(e)
    return env


def choose_pytorch_index_url(nvidia_info: dict) -> list[str]:
    # Prefer CUDA 12.1 wheels; fallback to 11.8 if needed
    urls: list[str] = [
        "https://download.pytorch.org/whl/cu121",
        "https://download.pytorch.org/whl/cu118",
    ]
    # If nvidia-smi reports CUDA 11.x, try cu118 first
    try:
        cuda_txt = nvidia_info.get("cuda_version") or ""
        major_minor = float("".join(re.findall(r"^([0-9]+\.[0-9]+)", cuda_txt))) if cuda_txt else 0.0
        if 11.0 <= major_minor < 12.0:
            urls = ["https://download.pytorch.org/whl/cu118", "https://download.pytorch.org/whl/cu121"]
    except Exception:
        pass
    return urls


def install_pytorch_cuda(index_url: str) -> Tuple[bool, str]:
    # Install torch and torchvision from a specific CUDA wheel index
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--user",
        "--index-url",
        index_url,
        "torch",
        "torchvision",
    ]
    code, out, err = run(cmd, timeout=1800)
    success = code == 0
    return success, out + ("\n" + err if err else "")


def main() -> int:
    print("===== GPU PREFLIGHT START =====")
    print(f"Python: {sys.version.replace('\n', ' ')}")
    print(f"Executable: {sys.executable}")

    has_gpu, nv_info = detect_nvidia_gpu()
    print(f"NVIDIA GPU Present: {has_gpu}")
    print(f"NVIDIA Info: {json.dumps(nv_info)}")

    env_before = torch_env()
    print(f"Torch (before): {json.dumps(env_before)}")

    if not has_gpu:
        print("No NVIDIA GPU detected via nvidia-smi. Skipping CUDA install.")
        print("===== GPU PREFLIGHT END (NO GPU) =====")
        return 0

    # If torch already has CUDA and reports is_available, we are done
    if env_before.get("torch_imported") and env_before.get("torch_cuda_version") and env_before.get("cuda_is_available"):
        print("PyTorch already has CUDA support and is available.")
        print("===== GPU PREFLIGHT END (OK) =====")
        return 0

    # Try installing CUDA-enabled PyTorch
    for url in choose_pytorch_index_url(nv_info):
        print(f"Attempting to install CUDA-enabled PyTorch from: {url}")
        ok, log = install_pytorch_cuda(url)
        print(log)
        if ok:
            env_after = torch_env()
            print(f"Torch (after): {json.dumps(env_after)}")
            if env_after.get("torch_imported") and env_after.get("torch_cuda_version") and env_after.get("cuda_is_available"):
                print("CUDA-enabled PyTorch installation verified.")
                print("===== GPU PREFLIGHT END (FIXED) =====")
                return 0
        else:
            print(f"Install from {url} failed; trying next option if available...")

    print("Failed to enable CUDA for PyTorch. Backend will run on CPU.")
    print("Hints: Ensure a recent NVIDIA driver is installed and internet access is available.")
    print("===== GPU PREFLIGHT END (FAILED) =====")
    return 0  # Do not block app startup


if __name__ == "__main__":
    sys.exit(main())


