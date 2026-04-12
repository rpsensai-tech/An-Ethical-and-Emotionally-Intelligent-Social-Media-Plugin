"""
Device Manager
Handles device selection and management for PyTorch models
"""

import torch
from typing import Optional, Dict, Any
import logging


class DeviceManager:
    """Manages device selection and torch operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._available_devices = self._detect_devices()
    
    def _detect_devices(self) -> Dict[str, Dict[str, Any]]:
        """Detect available computing devices"""
        devices = {
            "cpu": {
                "available": True,
                "name": "CPU",
                "memory": self._get_cpu_memory(),
                "recommended": True
            }
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                devices[f"cuda:{i}"] = {
                    "available": True,
                    "name": device_props.name,
                    "memory": device_props.total_memory // (1024**3),  # GB
                    "compute_capability": f"{device_props.major}.{device_props.minor}",
                    "recommended": device_props.total_memory > 4 * (1024**3)  # 4GB+
                }
            
            devices["cuda"] = devices["cuda:0"] if "cuda:0" in devices else {
                "available": True,
                "name": "CUDA Device",
                "memory": 0,
                "recommended": False
            }
        else:
            devices["cuda"] = {
                "available": False,
                "name": "CUDA Not Available",
                "memory": 0,
                "recommended": False
            }
        
        return devices
    
    def _get_cpu_memory(self) -> int:
        """Get available CPU memory in GB"""
        try:
            import psutil
            return psutil.virtual_memory().total // (1024**3)
        except ImportError:
            return 8  # Default assumption
    
    def get_available_devices(self) -> Dict[str, Dict[str, Any]]:
        """Get all available devices with their properties"""
        return self._available_devices
    
    def get_device(self, requested_device: str = "auto") -> torch.device:
        """Get torch device based on request and availability"""
        if requested_device == "auto":
            if self._available_devices.get("cuda", {}).get("available", False):
                device = torch.device("cuda")
                self.logger.info(f"Auto-selected CUDA device: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                self.logger.info("Auto-selected CPU device")
            return device
        
        if requested_device == "cpu":
            return torch.device("cpu")
        
        if requested_device.startswith("cuda"):
            if not torch.cuda.is_available():
                self.logger.warning("CUDA requested but not available, falling back to CPU")
                return torch.device("cpu")
            
            try:
                device = torch.device(requested_device)
                torch.tensor([1.0]).to(device)
                return device
            except RuntimeError as e:
                self.logger.warning(f"Failed to use {requested_device}: {e}, falling back to CPU")
                return torch.device("cpu")
        
        self.logger.warning(f"Invalid device '{requested_device}', falling back to CPU")
        return torch.device("cpu")
    
    def get_memory_info(self, device: torch.device) -> Dict[str, Any]:
        """Get memory information for the device"""
        if device.type == "cuda":
            if torch.cuda.is_available():
                return {
                    "total": torch.cuda.get_device_properties(device).total_memory,
                    "allocated": torch.cuda.memory_allocated(device),
                    "cached": torch.cuda.memory_reserved(device)
                }
        return {"total": 0, "allocated": 0, "cached": 0}
    
    def clear_memory(self, device: torch.device):
        """Clear GPU memory cache"""
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("Cleared GPU memory cache")


# Global instance
device_manager = DeviceManager()
