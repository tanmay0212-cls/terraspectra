"""
Hardware and Environment Diagnostics for TerraSpectra.

Distinguishes between detected physical hardware and verified usable software acceleration.
Safely handles operating system restrictions (such as Windows Smart App Control) without crashing.
"""

from __future__ import annotations
import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional
import psutil


def get_cpu_info() -> Dict[str, Any]:
    """Retrieve CPU architecture and core counts."""
    return {
        "processor": platform.processor(),
        "architecture": platform.machine(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
    }


def get_memory_info() -> Dict[str, Any]:
    """Retrieve host RAM details."""
    vm = psutil.virtual_memory()
    return {
        "total_gb": round(vm.total / (1024 ** 3), 2),
        "available_gb": round(vm.available / (1024 ** 3), 2),
        "percent_used": vm.percent,
    }


def get_disk_info(path: str = ".") -> Dict[str, Any]:
    """Retrieve disk capacity for the active workspace."""
    usage = shutil.disk_usage(path)
    return {
        "total_gb": round(usage.total / (1024 ** 3), 2),
        "free_gb": round(usage.free / (1024 ** 3), 2),
        "used_gb": round(usage.used / (1024 ** 3), 2),
    }


def detect_nvidia_gpu() -> Dict[str, Any]:
    """
    Detect NVIDIA GPU hardware via nvidia-smi if accessible.
    Reports raw hardware presence, NOT verified execution capability.
    """
    gpu_info = {
        "detected": False,
        "name": None,
        "driver_version": None,
        "total_vram_mb": None,
        "details": "nvidia-smi not available or no NVIDIA device found"
    }
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            check=False
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            if len(parts) >= 3:
                gpu_info["detected"] = True
                gpu_info["name"] = parts[0]
                gpu_info["driver_version"] = parts[1]
                try:
                    gpu_info["total_vram_mb"] = float(parts[2])
                except ValueError:
                    gpu_info["total_vram_mb"] = parts[2]
                gpu_info["details"] = "Hardware detected via nvidia-smi"
    except Exception as e:
        gpu_info["details"] = f"Detection error: {type(e).__name__}: {e}"
    return gpu_info


def inspect_pytorch_runtime() -> Dict[str, Any]:
    """
    Safely inspect PyTorch import capability without crashing on OS permission or DLL errors.
    """
    status = {
        "installed": False,
        "importable": False,
        "version": None,
        "cuda_available": False,
        "device_name": None,
        "error_message": None,
    }
    try:
        # Dynamic import wrapped in try-except
        import torch  # type: ignore
        status["installed"] = True
        status["importable"] = True
        status["version"] = getattr(torch, "__version__", "unknown")
        status["cuda_available"] = torch.cuda.is_available()
        if status["cuda_available"]:
            status["device_name"] = torch.cuda.get_device_name(0)
    except OSError as oe:
        # Check for Windows Smart App Control / AppLocker WinError 4551
        status["installed"] = True
        status["importable"] = False
        if "4551" in str(oe) or "Application Control" in str(oe):
            status["error_message"] = (
                "Blocked by Windows Smart App Control (SAC) / Application Control policy (WinError 4551). "
                "PyTorch dynamic library (c10.dll) could not be loaded."
            )
        else:
            status["error_message"] = f"OS/DLL load error: {oe}"
    except ImportError as ie:
        status["installed"] = False
        status["importable"] = False
        status["error_message"] = f"PyTorch is not installed: {ie}"
    except Exception as e:
        status["installed"] = True
        status["importable"] = False
        status["error_message"] = f"Unexpected error importing torch: {type(e).__name__}: {e}"

    return status


def inspect_onnxruntime() -> Dict[str, Any]:
    """Inspect installed ONNX Runtime version and available execution providers."""
    status = {
        "installed": False,
        "version": None,
        "available_providers": [],
        "cpu_provider_verified": False,
        "acceleration_notes": "None"
    }
    try:
        import onnxruntime as ort
        status["installed"] = True
        status["version"] = getattr(ort, "__version__", "unknown")
        providers = ort.get_available_providers()
        status["available_providers"] = providers
        status["cpu_provider_verified"] = "CPUExecutionProvider" in providers
        notes = []
        if "CPUExecutionProvider" in providers:
            notes.append("Verified CPU execution ready")
        if "DmlExecutionProvider" in providers:
            notes.append("DirectML provider registered (requires test session verification)")
        if "CUDAExecutionProvider" in providers:
            notes.append("CUDA provider registered (requires test session verification)")
        status["acceleration_notes"] = "; ".join(notes) if notes else "No providers registered"
    except Exception as e:
        status["acceleration_notes"] = f"Failed to inspect onnxruntime: {e}"
    return status


def inspect_installed_packages() -> Dict[str, Optional[str]]:
    """Inspect core scientific, ML, and geospatial package versions."""
    packages = [
        "numpy",
        "scipy",
        "pandas",
        "sklearn",
        "cv2",
        "tifffile",
        "fastapi",
        "uvicorn",
        "pydantic",
        "pytest",
        "rasterio",
        "osgeo",  # GDAL
        "geopandas",
        "pydeck",
        "yaml",
    ]
    report = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            report[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            report[pkg] = None
        except Exception as e:
            report[pkg] = f"Error: {e}"
    return report


def run_environment_diagnostics() -> Dict[str, Any]:
    """Run full diagnostic audit of the execution environment."""
    pytorch_info = inspect_pytorch_runtime()
    onnx_info = inspect_onnxruntime()
    gpu_info = detect_nvidia_gpu()

    # Determine verified usable acceleration
    verified_acceleration = "CPU (Single/Multithreaded via NumPy/Scipy/ONNX-CPU)"
    if onnx_info.get("cpu_provider_verified"):
        verified_acceleration += " + ONNX Runtime CPU"
    
    hardware_acceleration_detected = []
    if gpu_info.get("detected"):
        hardware_acceleration_detected.append(f"GPU: {gpu_info.get('name')} ({gpu_info.get('total_vram_mb')} MB VRAM)")
    
    return {
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "platform": sys.platform,
            "os": f"{platform.system()} {platform.release()} ({platform.version()})"
        },
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "disk": get_disk_info(),
        "gpu_hardware": gpu_info,
        "pytorch": pytorch_info,
        "onnxruntime": onnx_info,
        "verified_usable_acceleration": verified_acceleration,
        "hardware_detected_but_unverified": hardware_acceleration_detected,
        "installed_packages": inspect_installed_packages(),
    }


def format_diagnostics_report(diag: Dict[str, Any]) -> str:
    """Format diagnostic dictionary into clean, human-readable markdown text."""
    lines = [
        "# TerraSpectra Environment Diagnostics Report",
        "",
        "## System & Hardware",
        f"- **OS:** {diag['python']['os']}",
        f"- **Python:** {diag['python']['version']} ({diag['python']['executable']})",
        f"- **CPU:** {diag['cpu']['processor']} ({diag['cpu']['physical_cores']} phys / {diag['cpu']['logical_cores']} log cores, {diag['cpu']['architecture']})",
        f"- **RAM:** {diag['memory']['total_gb']} GB total ({diag['memory']['available_gb']} GB available, {diag['memory']['percent_used']}% used)",
        f"- **Workspace Disk:** {diag['disk']['total_gb']} GB total ({diag['disk']['free_gb']} GB free)",
        "",
        "## Acceleration & Runtimes",
        f"- **Hardware GPU Detected:** {diag['gpu_hardware']['name']} ({diag['gpu_hardware'].get('total_vram_mb')} MB VRAM)" if diag['gpu_hardware']['detected'] else "- **Hardware GPU Detected:** None",
        f"- **Verified Usable Acceleration:** {diag['verified_usable_acceleration']}",
        "",
        "## PyTorch Status",
        f"- **Installed:** {diag['pytorch']['installed']}",
        f"- **Importable:** {diag['pytorch']['importable']}",
        f"- **Version:** {diag['pytorch']['version']}",
        f"- **Status Notes:** {diag['pytorch']['error_message'] or 'Operational'}",
        "",
        "## ONNX Runtime Status",
        f"- **Installed:** {diag['onnxruntime']['installed']} (v{diag['onnxruntime']['version']})",
        f"- **Available Providers:** {', '.join(diag['onnxruntime']['available_providers'])}",
        f"- **CPU Provider Verified:** {diag['onnxruntime']['cpu_provider_verified']}",
        f"- **Provider Notes:** {diag['onnxruntime']['acceleration_notes']}",
        "",
        "## Core Packages",
    ]
    for pkg, ver in diag["installed_packages"].items():
        state = f"v{ver}" if ver and not ver.startswith("Error") else (ver or "NOT INSTALLED")
        lines.append(f"- **{pkg}:** {state}")

    return "\n".join(lines)
