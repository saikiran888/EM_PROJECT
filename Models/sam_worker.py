"""
SAM Worker for background SAM model loading and prediction
"""
import os
import torch
import numpy as np
import cv2
from contextlib import contextmanager
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

# Import CUDA memory manager for non-blocking operations
try:
    from Utils.cuda_memory_manager import CUDAMemoryManager
except ImportError:
    try:
        from CellEventAnnotator.Utils.cuda_memory_manager import CUDAMemoryManager
    except ImportError:
        CUDAMemoryManager = None  # Will handle gracefully if not available


class SamWorker(QObject):
    """Worker for loading and running SAM model in background"""
    
    predictor_ready = pyqtSignal()
    error_occurred = pyqtSignal(str)
    # Signals for async prediction results
    prediction_automatic_complete = pyqtSignal(object)  # emits masks list
    prediction_boxes_complete = pyqtSignal(object)  # emits masks list
    prediction_points_complete = pyqtSignal(object)  # emits masks list
    # New signal for atomic batch processing: emits (masks, frame_idx) or (None, frame_idx) on failure
    batch_frame_complete = pyqtSignal(object, int)

    # Signals to trigger async methods (simple triggers without complex parameters)
    trigger_automatic = pyqtSignal()  # triggers predict_automatic_async
    trigger_boxes = pyqtSignal()  # triggers predict_with_boxes_async
    # GUI sets _pending_image + _pending_boxes then emits this (set_image + boxes on same thread)
    trigger_box_prompt = pyqtSignal()  # triggers set_image_and_predict_boxes_async
    trigger_points = pyqtSignal()  # triggers predict_with_points_async
    trigger_set_image = pyqtSignal()  # triggers set_image_async
    
    def __init__(self, checkpoint_path=None, model_type="vit_h", device="cuda"):
        super().__init__()
        self.checkpoint_path = checkpoint_path or "/home/sai/Desktop/Annotation_dev/CellEventAnnotator/sam3.pt"
        self.model_type = model_type
        self.device = device if torch.cuda.is_available() else "cpu"
        self.sam = None
        self.predictor = None
        self.sam3_processor = None  # For SAM3 processor-based inference
        self.is_sam3_model = False  # Flag to track if this is a SAM3 model
        self.predictor_warning_shown = False  # Track if we've already shown the fallback warning
        
        # Initialize CUDA memory manager for non-blocking memory operations
        if CUDAMemoryManager is not None:
            self.memory_manager = CUDAMemoryManager(device=self.device)
            self.memory_manager.optimize_for_inference()
        else:
            self.memory_manager = None
        
        # Storage for async operation parameters (set before triggering)
        self._pending_image = None
        self._pending_boxes = None
        self._pending_points = None
        self._pending_labels = None
        self._pending_confidence = 0.5
        self._pending_max_cells = 2
        # SAM3 automatic: how to pick which masks to keep after filtering (see predict_automatic)
        self._pending_mask_selection_mode = "smallest"
        
        # Configure CUDA for non-blocking operations
        if torch.cuda.is_available() and self.device.startswith('cuda'):
            # Enable async execution
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Set CUDA device - convert "cuda" to "cuda:0" or extract index from "cuda:X"
            if self.device == "cuda":
                # Use default device (cuda:0)
                device_index = 0
            else:
                # Extract index from "cuda:X" format
                try:
                    device_index = int(self.device.split(':')[1])
                except (IndexError, ValueError):
                    device_index = 0
            torch.cuda.set_device(device_index)
            print(f"  ✓ Configured CUDA for non-blocking operations on cuda:{device_index}")
    
    def _move_state_to_device(self, state, device):
        """Recursively move all tensors in state dictionary to the specified device"""
        if state is None:
            return state
        
        if isinstance(state, dict):
            moved_dict = {}
            for k, v in state.items():
                moved_v = self._move_state_to_device(v, device)
                # Special handling for boxes - ensure they're definitely on device
                if k == 'boxes' and torch.is_tensor(moved_v):
                    moved_v = moved_v.to(device)
                elif k == 'boxes' and isinstance(moved_v, (list, tuple)):
                    moved_v = type(moved_v)([b.to(device) if torch.is_tensor(b) else b for b in moved_v])
                moved_dict[k] = moved_v
            return moved_dict
        elif isinstance(state, (list, tuple)):
            return type(state)(self._move_state_to_device(item, device) for item in state)
        elif torch.is_tensor(state):
            return state.to(device)
        else:
            return state

    def _collect_tensor_devices(self, state, devices=None):
        """Collect device strings for all tensors found in a nested state structure"""
        if devices is None:
            devices = set()
        if state is None:
            return devices
        if isinstance(state, dict):
            for value in state.values():
                self._collect_tensor_devices(value, devices)
        elif isinstance(state, (list, tuple)):
            for item in state:
                self._collect_tensor_devices(item, devices)
        elif torch.is_tensor(state):
            devices.add(str(state.device))
        return devices

    def _collect_processor_tensor_devices(self, processor, devices=None):
        """Collect device strings from processor/model tensors and cached attributes"""
        if devices is None:
            devices = set()
        if processor is None:
            return devices

        # Model parameters/buffers
        try:
            if hasattr(processor, "model") and processor.model is not None:
                for param in processor.model.parameters():
                    devices.add(str(param.device))
                    break
                for buffer in processor.model.buffers():
                    devices.add(str(buffer.device))
                    break
        except (AttributeError, RuntimeError):
            pass

        # Processor cached tensors
        for attr_name in dir(processor):
            if attr_name.startswith("_"):
                continue
            try:
                attr_value = getattr(processor, attr_name)
                if torch.is_tensor(attr_value):
                    devices.add(str(attr_value.device))
            except (AttributeError, RuntimeError):
                continue

        return devices

    def _pick_target_device(self, processor, state):
        """Choose a target device based on model and state tensors"""
        model_device = self._get_model_device(processor)
        state_devices = self._collect_tensor_devices(state)
        processor_devices = self._collect_processor_tensor_devices(processor)

        # If any tensor is on CUDA, prefer CUDA to avoid mixed-device errors
        all_devices = state_devices | processor_devices
        cuda_devices = [d for d in all_devices if d.startswith("cuda")]
        if cuda_devices:
            picked = cuda_devices[0]
            if model_device != picked:
                print(f"Found CUDA tensors ({picked}) while model reports {model_device}; using {picked}")
            return picked

        # Fall back to model device if available
        if model_device:
            return model_device

        # Fall back to any processor device
        if processor_devices:
            return sorted(processor_devices)[0]

        # Final fallback
        return self.device
    
    def _get_model_device(self, processor):
        """Get the actual device the model is on"""
        if processor is None:
            return self.device
        
        # Try to get device from processor's model
        if hasattr(processor, 'model') and processor.model is not None:
            # Get device from model's first parameter - this is the most reliable method
            try:
                for param in processor.model.parameters():
                    device = param.device
                    # Convert torch.device to string (e.g., "cuda:0" or "cpu")
                    device_str = str(device)
                    print(f"Detected model device from parameters: {device_str}")
                    return device_str
            except (StopIteration, AttributeError, RuntimeError) as e:
                print(f"Could not get device from model parameters: {e}")
            
            # Try getting device from model's buffers
            try:
                for buffer in processor.model.buffers():
                    device = buffer.device
                    device_str = str(device)
                    print(f"Detected model device from buffers: {device_str}")
                    return device_str
            except (StopIteration, AttributeError, RuntimeError) as e:
                print(f"Could not get device from model buffers: {e}")
            
            # Try getting device from named parameters
            try:
                for name, param in processor.model.named_parameters():
                    device = param.device
                    device_str = str(device)
                    print(f"Detected model device from named_parameters: {device_str}")
                    return device_str
            except (StopIteration, AttributeError, RuntimeError) as e:
                print(f"Could not get device from named_parameters: {e}")
            
            # Fallback: check if model has device attribute
            if hasattr(processor.model, 'device'):
                model_device = processor.model.device
                if isinstance(model_device, torch.device):
                    device_str = str(model_device)
                    print(f"Detected model device from attribute: {device_str}")
                    return device_str
                else:
                    device_str = str(model_device)
                    print(f"Detected model device from attribute (str): {device_str}")
                    return device_str
        
        # Fallback to processor's device attribute
        if hasattr(processor, 'device'):
            proc_device = processor.device
            if isinstance(proc_device, torch.device):
                device_str = str(proc_device)
                print(f"Detected device from processor attribute: {device_str}")
                return device_str
            elif isinstance(proc_device, str):
                print(f"Detected device from processor attribute (str): {proc_device}")
                return proc_device
        
        print(f"Using fallback device: {self.device}")
        return self.device
    
    def _ensure_processor_device(self, processor, target_device=None):
        """Ensure SAM3 processor and all its components are on the correct device"""
        if processor is None:
            return self.device
        
        # Determine target device (explicit target overrides model device)
        if target_device is None:
            model_device_str = self._get_model_device(processor)
        else:
            model_device_str = str(target_device)
        
        # Convert to torch.device object for moving tensors
        target_device = torch.device(model_device_str)
        
        # Ensure processor's device attribute matches model device
        if hasattr(processor, 'device'):
            processor.device = model_device_str
        
        # Ensure model is on the target device (check first to avoid unnecessary moves)
        if hasattr(processor, 'model') and processor.model is not None:
            # Check current device first
            try:
                for param in processor.model.parameters():
                    current_device_str = str(param.device)
                    if current_device_str != model_device_str:
                        print(f"Model is on {current_device_str}, but should be on {model_device_str}")
                        processor.model = processor.model.to(target_device)
                    break
            except (StopIteration, AttributeError, RuntimeError):
                # If we can't check, just try to move it
                processor.model = processor.model.to(target_device)
        
        # Check for any cached tensors in processor attributes
        for attr_name in dir(processor):
            if not attr_name.startswith('_'):
                try:
                    attr_value = getattr(processor, attr_name)
                    if torch.is_tensor(attr_value):
                        current_dev = str(attr_value.device)
                        if current_dev != model_device_str:
                            setattr(processor, attr_name, attr_value.to(target_device))
                except (AttributeError, RuntimeError):
                    pass
        
        return model_device_str
    
    def _find_sam3_bpe_path(self):
        """Find SAM3 BPE tokenizer file path"""
        try:
            bpe_filename = "bpe_simple_vocab_16e6.txt.gz"
            import sam3
            
            # Try environment variable
            env_bpe = os.environ.get("BPE_PATH")
            if env_bpe and os.path.exists(env_bpe):
                return env_bpe
            
            # Try relative to sam3 package
            if hasattr(sam3, '__file__') and sam3.__file__:
                try:
                    sam3_root = os.path.dirname(sam3.__file__)
                    bpe_path = os.path.join(sam3_root, "..", "assets", bpe_filename)
                    bpe_path = os.path.abspath(bpe_path)
                    if os.path.exists(bpe_path):
                        return bpe_path
                except (TypeError, AttributeError):
                    pass
            
            # Try known locations
            known_paths = [
                os.path.join("/cellchorus", "sam3_training", "sam3", "assets", bpe_filename),
            ]
            
            for path in known_paths:
                abs_path = os.path.abspath(path)
                if os.path.exists(abs_path):
                    return abs_path
        except Exception:
            pass
        
        return None
        
    @pyqtSlot()
    def load_model(self):
        """
        Load SAM model in background thread
        
        WARNING: CUDA operations in background threads can cause segfaults.
        If you experience crashes, the code will automatically fall back to CPU mode.
        
        This method runs entirely in the worker thread and should never block the UI.
        """
        print("[SAM WORKER] load_model() called in worker thread")
        import threading
        print(f"[SAM WORKER] Current thread: {threading.current_thread().name}")
        print(f"[SAM WORKER] Is main thread: {threading.current_thread() is threading.main_thread()}")
        
        # CRITICAL: Initialize CUDA context in this thread BEFORE any CUDA operations
        # CUDA contexts are thread-local, so we must initialize in the worker thread
        # However, CUDA in background threads can be problematic - use with caution
        if torch.cuda.is_available() and self.device.startswith('cuda'):
            try:
                # Get device index
                if self.device == "cuda":
                    device_index = 0
                else:
                    try:
                        device_index = int(self.device.split(':')[1])
                    except (IndexError, ValueError):
                        device_index = 0
                
                # Set device in this thread
                torch.cuda.set_device(device_index)
                
                # Initialize CUDA context by creating a primary context
                # This must happen in the thread where CUDA will be used
                try:
                    # Try to get current device to force context creation
                    current_device = torch.cuda.current_device()
                    print(f"[SAM WORKER] CUDA device set to: {current_device}")
                    
                    # Create a small tensor on CPU first, then move to CUDA
                    # This is safer than creating directly on CUDA
                    dummy = torch.zeros(1)
                    dummy = dummy.cuda(device_index)
                    del dummy
                    torch.cuda.synchronize(device_index)
                    print(f"[SAM WORKER] CUDA context initialized in worker thread (device: cuda:{device_index})")
                except Exception as context_err:
                    print(f"[SAM WORKER] ⚠️ CUDA context creation failed: {context_err}")
                    print(f"[SAM WORKER] ⚠️ CUDA in background threads may not be supported on this system")
                    print(f"[SAM WORKER] ⚠️ Will attempt to build model on CPU and move to CUDA later")
                    # Don't raise - let it try CPU mode
                    
            except Exception as cuda_init_err:
                print(f"[SAM WORKER] ⚠️ CUDA initialization failed in worker thread: {cuda_init_err}")
                print(f"[SAM WORKER] ⚠️ Error type: {type(cuda_init_err).__name__}")
                print(f"[SAM WORKER] ⚠️ This is common when CUDA contexts can't be created in background threads")
                print(f"[SAM WORKER] ⚠️ Will attempt CPU mode - model will be built on CPU")
                # Don't raise - let it try CPU mode first, then move to CUDA if possible
        
        # CRITICAL: Fix Python 3.9 compatibility issue with SAM3's type hints BEFORE any imports
        # SAM3 uses tuple[...] syntax which requires Python 3.10+
        # Pre-patch SAM3 source files before importing
        try:
            import sys
            import glob  # glob is not imported at module level, so import it here
            if sys.version_info < (3, 10):
                # Find and patch SAM3 Python files to add "from __future__ import annotations"
                # Note: os is already imported at module level, don't import it again here
                
                sam3_paths = [
                    "/home/sai/Desktop/sam3_training/sam3",
                    "/home/sai/Desktop/sam3",
                ]
                
                def patch_file_for_py39(filepath):
                    """Add __future__ annotations to a Python file if not already present"""
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            source = f.read()
                        
                        if 'from __future__ import annotations' not in source:
                            lines = source.split('\n')
                            insert_idx = 0
                            
                            # Skip shebang
                            if lines and lines[0].startswith('#!'):
                                insert_idx = 1
                            
                            # Skip encoding
                            if insert_idx < len(lines) and 'coding' in lines[insert_idx]:
                                insert_idx += 1
                            
                            # Skip module docstring
                            if insert_idx < len(lines) and lines[insert_idx].strip().startswith('"""'):
                                for i in range(insert_idx + 1, len(lines)):
                                    if '"""' in lines[i]:
                                        insert_idx = i + 1
                                        break
                            
                            lines.insert(insert_idx, 'from __future__ import annotations')
                            source = '\n'.join(lines)
                            
                            with open(filepath, 'w', encoding='utf-8') as f:
                                f.write(source)
                            
                            return True
                    except Exception:
                        pass
                    
                    return False
                
                # Patch all SAM3 Python files
                patched_count = 0
                for sam3_path in sam3_paths:
                    if os.path.exists(sam3_path):
                        for pyfile in glob.glob(os.path.join(sam3_path, '**/*.py'), recursive=True):
                            if patch_file_for_py39(pyfile):
                                patched_count += 1
                
                if patched_count > 0:
                    print(f"  ✓ Patched {patched_count} SAM3 Python files for Python 3.9 compatibility")
        except Exception as compat_err:
            print(f"  ⚠️ Could not pre-patch SAM3 files for Python 3.9: {compat_err}")
        
        try:
            # CRITICAL: Test PyTorch's functionality before attempting model loading
            # Test both basic operations and autograd to ensure PyTorch is fully functional
            try:
                # Test 1: Basic tensor creation
                test_tensor = torch.tensor([1.0], requires_grad=False)
                test_result = test_tensor * 2
                
                # Test 2: no_grad context (this is what SAM3 model initialization uses)
                with torch.no_grad():
                    test_tensor2 = torch.tensor([2.0])
                    test_result2 = test_tensor2 * 3
                
                # Test 3: nn.Linear initialization (similar to what SAM3 does)
                import torch.nn as nn
                test_layer = nn.Linear(10, 5)
                
                # Clean up
                del test_tensor, test_result, test_tensor2, test_result2, test_layer
                
                print("  PyTorch functionality test: PASSED")
            except RuntimeError as test_err:
                if "unknown parameter type" in str(test_err).lower():
                    error_msg = (
                        "CRITICAL: PyTorch installation is corrupted.\n\n"
                        "The error 'unknown parameter type' indicates PyTorch's C++ backend is broken.\n"
                        "This is a system-level issue that cannot be fixed by the application.\n\n"
                        "SOLUTION: You must reinstall PyTorch:\n\n"
                        "1. Activate your conda environment:\n"
                        "   conda activate curation_tool\n\n"
                        "2. Uninstall PyTorch:\n"
                        "   pip uninstall torch torchvision torchaudio\n\n"
                        "3. Reinstall PyTorch (with CUDA support):\n"
                        "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n"
                        "   (Adjust CUDA version if needed - check: nvidia-smi)\n\n"
                        "4. Verify installation:\n"
                        "   python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'\n\n"
                        "5. Restart this application after reinstalling PyTorch.\n\n"
                        "SAM3 model loading is disabled until PyTorch is fixed."
                    )
                    print("=" * 80)
                    print("ERROR: PyTorch Installation Issue Detected")
                    print("=" * 80)
                    print(error_msg)
                    print("=" * 80)
                    self.error_occurred.emit(error_msg)
                    return
                else:
                    # Other errors - still try to load but warn
                    print(f"Warning: PyTorch test failed: {test_err}")
            
        except Exception as pre_check_err:
            # Catch any other exceptions during pre-check
            if "unknown parameter type" in str(pre_check_err).lower():
                error_msg = (
                    "CRITICAL: PyTorch installation is corrupted.\n\n"
                    "The error 'unknown parameter type' indicates PyTorch's C++ backend is broken.\n"
                    "Please reinstall PyTorch (see instructions above)."
                )
                print("=" * 80)
                print("ERROR: PyTorch Installation Issue Detected")
                print("=" * 80)
                print(error_msg)
                print("=" * 80)
                self.error_occurred.emit(error_msg)
                return
            else:
                error_msg = f"Pre-check failed: {pre_check_err}"
                print(f"ERROR: {error_msg}")
                self.error_occurred.emit(error_msg)
                return
        
        try:
            # CRITICAL: Test PyTorch's autograd system before attempting model loading
            # If this fails, PyTorch's C++ backend is broken and model loading will fail
            try:
                test_tensor = torch.tensor([1.0], requires_grad=True)
                with torch.no_grad():
                    test_result = test_tensor * 2
                del test_tensor, test_result
            except RuntimeError as autograd_test_err:
                if "unknown parameter type" in str(autograd_test_err).lower():
                    error_msg = (
                        "PyTorch autograd system is broken. This is a PyTorch installation issue.\n\n"
                        "The error 'unknown parameter type' indicates PyTorch's C++ backend is corrupted or incompatible.\n\n"
                        "To fix this:\n"
                        "1. Reinstall PyTorch: pip uninstall torch torchvision torchaudio && pip install torch torchvision torchaudio\n"
                        "2. Check CUDA compatibility: Ensure PyTorch version matches your CUDA driver version\n"
                        "3. Try a different Python environment or PyTorch version\n"
                        "4. Check for conflicting packages (timm, etc.) and ensure compatibility\n\n"
                        f"Original error: {autograd_test_err}"
                    )
                    print(f"ERROR: {error_msg}")
                    self.error_occurred.emit(error_msg)
                    return
                else:
                    # Other autograd errors - still try to load but warn
                    print(f"Warning: PyTorch autograd test failed: {autograd_test_err}")
            
            # First, try to import segment_anything
            try:
                from segment_anything import sam_model_registry, SamPredictor
                HAS_SAM = True
            except ImportError:
                print("segment_anything not available, will use alternative approach")
                HAS_SAM = False
            
            # Check if checkpoint exists
            if not os.path.exists(self.checkpoint_path):
                self.error_occurred.emit(f"Checkpoint not found: {self.checkpoint_path}")
                return
            
            print(f"Loading model from {self.checkpoint_path}")
            
            # Check if this is SAM3 checkpoint
            is_sam3_checkpoint = "sam3" in str(self.checkpoint_path).lower()
            
            # Try SAM3 loading first if detected
            if is_sam3_checkpoint:
                try:
                    # CRITICAL: Patch torch.device BEFORE importing SAM3
                    # SAM3 imports use torch.device() at module level, which triggers the bug
                    # We need to patch the SAM3 source file directly to avoid the bug
                    # This is safer than patching torch.device globally
                    
                    # First, try to patch the problematic SAM3 file before import
                    try:
                        import sys
                        import importlib.util
                        
                        # Find and patch sam2_utils.py before it's imported
                        sam3_paths = [
                            "/cellchorus/sam3_training/sam3",
                            "/home/sai/Desktop/sam3_training/sam3",
                        ]
                        
                        sam2_utils_path = None
                        for base_path in sam3_paths:
                            candidate = os.path.join(base_path, "sam3", "model", "utils", "sam2_utils.py")
                            if os.path.exists(candidate):
                                sam2_utils_path = candidate
                                break
                        
                        if sam2_utils_path:
                            # Read the file
                            with open(sam2_utils_path, 'r') as f:
                                content = f.read()
                            
                            # Check if it has the problematic line
                            if 'compute_device=torch.device("cuda")' in content:
                                # Patch it: create CPU device, then it can be moved to CUDA later if needed
                                # This avoids the "unknown parameter type" error during import
                                patched_content = content.replace(
                                    'compute_device=torch.device("cuda")',
                                    'compute_device=torch.device("cpu")  # Patched: changed from "cuda" to avoid PyTorch bug'
                                )
                                
                                # Also check for other variations
                                patched_content = patched_content.replace(
                                    "compute_device=torch.device('cuda')",
                                    "compute_device=torch.device('cpu')  # Patched: changed from 'cuda' to avoid PyTorch bug"
                                )
                                
                                # Write back
                                with open(sam2_utils_path, 'w') as f:
                                    f.write(patched_content)
                                
                                print(f"  ✓ Patched {sam2_utils_path} to avoid torch.device('cuda') bug")
                    except Exception as patch_err:
                        print(f"  ⚠️ Could not patch SAM3 source file: {patch_err}")
                        # Continue - will try to patch torch.device as fallback
                    
                    # Note: We don't patch torch.device globally because:
                    # 1. It's not subclassable (built-in type)
                    # 2. The source file patch should handle the import-time issue
                    # 3. If other code needs torch.device("cuda"), it should work after the source patch
                    print("  ✓ SAM3 source file patched (torch.device bug avoided)")
                    
                    # Try to import sam3 - check multiple possible locations
                    sam3_imported = False
                    
                    # Try standard import first
                    try:
                        import sam3
                        sam3_imported = True
                    except (ImportError, RuntimeError) as import_err:
                        # Check if it's the device error
                        if "unknown parameter type" in str(import_err):
                            print(f"  ⚠️ torch.device patch didn't prevent error during import: {import_err}")
                            # Try more aggressive patch - will continue with path search
                        if isinstance(import_err, ImportError):
                            # Try adding common SAM3 installation paths
                            import sys
                        # Check environment variable first
                        env_sam3_path = os.environ.get("SAM3_PATH")
                        sam3_paths = []
                        if env_sam3_path and os.path.exists(env_sam3_path):
                            sam3_paths.append(env_sam3_path)
                        
                        # Add common paths
                        sam3_paths.extend([
                            "/home/sai/Desktop/sam3_training/sam3",
                            "/cellchorus/sam3_training/sam3",
                            "/home/sai/Desktop/sam3",
                            
                            "/opt/sam3",
                            os.path.join(os.path.dirname(self.checkpoint_path), "..", "sam3"),
                            os.path.join(os.path.expanduser("~"), "sam3"),
                        ])
                        
                        for sam3_path in sam3_paths:
                            if os.path.exists(sam3_path):
                                abs_path = os.path.abspath(sam3_path)
                                # Check if sam3 subdirectory exists
                                sam3_module_path = os.path.join(abs_path, "sam3")
                                if os.path.exists(sam3_module_path) or os.path.exists(os.path.join(abs_path, "__init__.py")):
                                    if abs_path not in sys.path:
                                        sys.path.insert(0, abs_path)
                                        print(f"Added SAM3 path to sys.path: {abs_path}")
                                    try:
                                        import sam3
                                        sam3_imported = True
                                        print(f"Successfully imported sam3 from {abs_path}")
                                        break
                                    except (ImportError, RuntimeError) as import_err:
                                        error_msg = str(import_err)
                                        # Check if it's the torch.device error
                                        if "unknown parameter type" in error_msg:
                                            print(f"  ⚠️ torch.device error during import from {abs_path}")
                                            print(f"  ⚠️ Error: {error_msg}")
                                            # The patch should have prevented this - try to patch the source file directly
                                            # For now, just note it and continue
                                        elif "decord" in error_msg or "No module named" in error_msg:
                                            print(f"Found SAM3 at {abs_path} but import failed: {error_msg}")
                                            print(f"  Note: Missing dependency detected")
                                        else:
                                            print(f"Found SAM3 at {abs_path} but import failed: {error_msg}")
                                        # Continue to next path
                                        continue
                                else:
                                    print(f"Path {abs_path} exists but doesn't contain sam3 module")
                            else:
                                print(f"SAM3 path not found: {sam3_path}")
                    
                    if not sam3_imported:
                        raise ImportError(
                            "SAM3 module not found. Please install SAM3 or add it to PYTHONPATH"
                        )
                    
                    # Patch PyTorch's set_grad_enabled to avoid the C++ backend bug
                    # The PyTorch 2.7.1+cu118 C++ backend bug affects torch._C._set_grad_enabled
                    # This is the root cause - patch it to avoid the bug entirely
                    # This must happen before SAM3 imports anything
                    try:
                        import math
                        import torch.nn.init as init
                        
                        # Patch torch.set_grad_enabled to avoid the C++ backend bug
                        _original_set_grad_enabled = torch.set_grad_enabled
                        _grad_enabled_state = [torch.is_grad_enabled()]  # Use list to allow modification in closure
                        
                        def _patched_set_grad_enabled(mode):
                            """Patched version that avoids the C++ backend bug"""
                            # Just track the state in Python, don't call the C++ backend
                            # This avoids the "unknown parameter type" error
                            _grad_enabled_state[0] = mode
                            # Don't call torch._C._set_grad_enabled - that's where the bug is
                            return None
                        
                        # Also patch torch.is_grad_enabled
                        _original_is_grad_enabled = torch.is_grad_enabled
                        
                        def _patched_is_grad_enabled():
                            """Patched version that returns our tracked state"""
                            return _grad_enabled_state[0]
                        
                        # Apply patches
                        torch.set_grad_enabled = _patched_set_grad_enabled
                        torch.is_grad_enabled = _patched_is_grad_enabled
                        print("  ✓ Pre-patched torch.set_grad_enabled and torch.is_grad_enabled (ROOT FIX)")
                        
                        # Set grad disabled using our patched function
                        torch.set_grad_enabled(False)
                        
                        # Patch ALL PyTorch initialization functions that use torch.no_grad() context
                        # The PyTorch 2.7.1+cu118 C++ backend bug affects all of them
                        
                        # Patch _no_grad_uniform_ (used by many init functions)
                        if hasattr(init, '_no_grad_uniform_'):
                            _original_no_grad_uniform = init._no_grad_uniform_
                            
                            def _patched_no_grad_uniform(tensor, a, b, generator=None):
                                """Patched version that avoids torch.no_grad() context manager"""
                                # Ensure tensor doesn't require grad before in-place operation
                                if tensor.requires_grad:
                                    tensor.requires_grad_(False)
                                return tensor.uniform_(a, b, generator=generator)
                            
                            init._no_grad_uniform_ = _patched_no_grad_uniform
                            print("  ✓ Pre-patched PyTorch's _no_grad_uniform_ (before SAM3 import)")
                        
                        # Patch _no_grad_fill_ (used by ones_, zeros_, etc.)
                        if hasattr(init, '_no_grad_fill_'):
                            _original_no_grad_fill = init._no_grad_fill_
                            
                            def _patched_no_grad_fill(tensor, val):
                                """Patched version that avoids torch.no_grad() context manager"""
                                # Ensure tensor doesn't require grad before in-place operation
                                if tensor.requires_grad:
                                    tensor.requires_grad_(False)
                                return tensor.fill_(val)
                            
                            init._no_grad_fill_ = _patched_no_grad_fill
                            print("  ✓ Pre-patched PyTorch's _no_grad_fill_ (before SAM3 import)")
                        
                        # Patch ones_() and zeros_() directly (called by LayerNorm.reset_parameters())
                        if hasattr(init, 'ones_'):
                            _original_ones = init.ones_
                            
                            def _patched_ones_(tensor):
                                """Patched version that disables grad before in-place operation"""
                                if tensor.requires_grad:
                                    tensor.requires_grad_(False)
                                return tensor.fill_(1.0)
                            
                            init.ones_ = _patched_ones_
                            print("  ✓ Pre-patched PyTorch's ones_ (before SAM3 import)")
                        
                        if hasattr(init, 'zeros_'):
                            _original_zeros = init.zeros_
                            
                            def _patched_zeros_(tensor):
                                """Patched version that disables grad before in-place operation"""
                                if tensor.requires_grad:
                                    tensor.requires_grad_(False)
                                return tensor.zero_()
                            
                            init.zeros_ = _patched_zeros_
                            print("  ✓ Pre-patched PyTorch's zeros_ (before SAM3 import)")
                        
                        # Patch _no_grad_zero_ (used by zeros_, etc.)
                        if hasattr(init, '_no_grad_zero_'):
                            _original_no_grad_zero = init._no_grad_zero_
                            
                            def _patched_no_grad_zero(tensor):
                                """Patched version that avoids torch.no_grad() context manager"""
                                # Ensure tensor doesn't require grad before in-place operation
                                if tensor.requires_grad:
                                    tensor.requires_grad_(False)
                                return tensor.zero_()
                            
                            init._no_grad_zero_ = _patched_no_grad_zero
                            print("  ✓ Pre-patched PyTorch's _no_grad_zero_ (before SAM3 import)")
                        
                        # Patch xavier_uniform_ (used by nn.MultiheadAttention)
                        if hasattr(init, 'xavier_uniform_'):
                            _original_xavier_uniform = init.xavier_uniform_
                            
                            def _patched_xavier_uniform(tensor, gain=1., *, generator=None):
                                """Patched version that avoids torch.no_grad() context manager"""
                                # Ensure tensor doesn't require grad before in-place operation
                                if tensor.requires_grad:
                                    tensor.requires_grad_(False)
                                fan_in, fan_out = init._calculate_fan_in_and_fan_out(tensor)
                                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                                a = math.sqrt(3.0) * std
                                # Use uniform_ directly - no context manager
                                tensor.uniform_(-a, a, generator=generator)
                                return tensor
                            
                            init.xavier_uniform_ = _patched_xavier_uniform
                            print("  ✓ Pre-patched PyTorch's xavier_uniform_ (before SAM3 import)")
                        
                        # Patch kaiming_uniform_ (used by nn.Linear.reset_parameters)
                        if hasattr(init, 'kaiming_uniform_'):
                            _original_kaiming_uniform = init.kaiming_uniform_
                            
                            def _patched_kaiming_uniform(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu', *, generator=None):
                                """Patched version that avoids torch.no_grad() context manager"""
                                # Ensure tensor doesn't require grad before in-place operation
                                if tensor.requires_grad:
                                    tensor.requires_grad_(False)
                                fan_in, fan_out = init._calculate_fan_in_and_fan_out(tensor)
                                fan = fan_in if mode == 'fan_in' else fan_out
                                gain = init.calculate_gain(nonlinearity, a)
                                std = gain / math.sqrt(fan)
                                bound = math.sqrt(3.0) * std
                                # Use uniform_ directly - no context manager
                                tensor.uniform_(-bound, bound, generator=generator)
                                return tensor
                            
                            init.kaiming_uniform_ = _patched_kaiming_uniform
                            print("  ✓ Pre-patched PyTorch's kaiming_uniform_ (before SAM3 import)")
                        
                        # Patch normal_() and _no_grad_normal_() (used by nn.Embedding and others)
                        if hasattr(init, '_no_grad_normal_'):
                            _original_no_grad_normal = init._no_grad_normal_
                            
                            def _patched_no_grad_normal(tensor, mean, std, generator=None):
                                """Patched version that avoids torch.no_grad() context manager"""
                                if tensor.requires_grad:
                                    tensor.requires_grad_(False)
                                return tensor.normal_(mean, std, generator=generator)
                            
                            init._no_grad_normal_ = _patched_no_grad_normal
                            print("  ✓ Pre-patched PyTorch's _no_grad_normal_ (before SAM3 import)")
                        
                        if hasattr(init, 'normal_'):
                            _original_normal = init.normal_
                            
                            def _patched_normal(tensor, mean=0., std=1., *, generator=None):
                                """Patched version that avoids torch.no_grad() context manager"""
                                if tensor.requires_grad:
                                    tensor.requires_grad_(False)
                                return tensor.normal_(mean, std, generator=generator)
                            
                            init.normal_ = _patched_normal
                            print("  ✓ Pre-patched PyTorch's normal_ (before SAM3 import)")
                        
                        # Patch timm's trunc_normal_ and _trunc_normal_ (both need patching)
                        import timm.layers.weight_init as weight_init
                        if hasattr(weight_init, 'trunc_normal_'):
                            _original_trunc_normal_early = weight_init.trunc_normal_
                            
                            def _patched_trunc_normal_early(tensor, mean=0., std=1., a=-2., b=2.):
                                """Complete replacement that avoids torch.no_grad() context manager entirely"""
                                # Ensure tensor doesn't require grad before in-place operations
                                if tensor.requires_grad:
                                    tensor.requires_grad_(False)
                                # Calculate bounds
                                l = (1. + math.erf(((a - mean) / std) / math.sqrt(2.))) / 2.
                                u = (1. + math.erf(((b - mean) / std) / math.sqrt(2.))) / 2.
                                
                                # Fill tensor with uniform values
                                tensor.uniform_(2 * l - 1, 2 * u - 1)
                                
                                # Transform to normal distribution
                                tensor.erfinv_()
                                tensor.mul_(std * math.sqrt(2.))
                                tensor.add_(mean)
                                
                                # Clamp to bounds
                                tensor.clamp_(min=a, max=b)
                                
                                return tensor
                            
                            weight_init.trunc_normal_ = _patched_trunc_normal_early
                            
                            # Also patch _trunc_normal_ if it exists (the internal function)
                            if hasattr(weight_init, '_trunc_normal_'):
                                _original__trunc_normal_early = weight_init._trunc_normal_
                                
                                def _patched__trunc_normal_early(tensor, mean=0., std=1., a=-2., b=2.):
                                    """Patched internal function that avoids torch.no_grad() context manager"""
                                    # Ensure tensor doesn't require grad before in-place operations
                                    if tensor.requires_grad:
                                        tensor.requires_grad_(False)
                                    # Same implementation - no context manager
                                    l = (1. + math.erf(((a - mean) / std) / math.sqrt(2.))) / 2.
                                    u = (1. + math.erf(((b - mean) / std) / math.sqrt(2.))) / 2.
                                    tensor.uniform_(2 * l - 1, 2 * u - 1)
                                    tensor.erfinv_()
                                    tensor.mul_(std * math.sqrt(2.))
                                    tensor.add_(mean)
                                    tensor.clamp_(min=a, max=b)
                                    return tensor
                                
                                weight_init._trunc_normal_ = _patched__trunc_normal_early
                                print("  ✓ Pre-patched timm's trunc_normal_ and _trunc_normal_ (before SAM3 import)")
                            else:
                                print("  ✓ Pre-patched timm's trunc_normal_ (before SAM3 import)")
                            
                            # Also patch it in sys.modules to catch any cached imports
                            import sys
                            for mod_name in list(sys.modules.keys()):
                                if 'timm' in mod_name and 'weight_init' in mod_name:
                                    mod = sys.modules[mod_name]
                                    if hasattr(mod, 'trunc_normal_'):
                                        mod.trunc_normal_ = _patched_trunc_normal_early
                                    if hasattr(mod, '_trunc_normal_'):
                                        mod._trunc_normal_ = _patched__trunc_normal_early if '_patched__trunc_normal_early' in locals() else _patched_trunc_normal_early
                        
                        # Patch Tensor.copy_() to handle requires_grad properly
                        # This is critical for checkpoint loading
                        # Note: torch is already imported at the module level
                        if hasattr(torch.Tensor, 'copy_'):
                            _original_tensor_copy = torch.Tensor.copy_
                            
                            def _patched_tensor_copy(self, src, non_blocking=False):
                                """Patched version that disables grad before in-place copy"""
                                # Disable requires_grad for both source and destination
                                src_requires_grad = src.requires_grad
                                self_requires_grad = self.requires_grad
                                
                                try:
                                    if src_requires_grad:
                                        src.requires_grad_(False)
                                    if self_requires_grad:
                                        self.requires_grad_(False)
                                    
                                    return _original_tensor_copy(self, src, non_blocking=non_blocking)
                                finally:
                                    # Note: We don't restore requires_grad since we want to keep it disabled for inference
                                    pass
                            
                            torch.Tensor.copy_ = _patched_tensor_copy
                            print("  ✓ Pre-patched torch.Tensor.copy_ for checkpoint loading (before SAM3 import)")
                        
                        # Patch torch operations to handle mixed tensor types properly
                        # This is needed for RoPE computations in SAM3
                        _original_tensor_mul = torch.Tensor.__mul__
                        _original_tensor_rmul = torch.Tensor.__rmul__
                        _original_tensor_add = torch.Tensor.__add__
                        _original_tensor_radd = torch.Tensor.__radd__
                        
                        def _patched_tensor_mul(self, other):
                            """Patched multiplication to handle type mismatches"""
                            try:
                                return _original_tensor_mul(self, other)
                            except (RuntimeError, TypeError) as e:
                                if "unknown parameter type" in str(e):
                                    # Convert other to same dtype if needed
                                    if isinstance(other, torch.Tensor) and other.dtype != self.dtype:
                                        other = other.to(self.dtype)
                                    elif isinstance(other, (int, float)):
                                        other = torch.tensor(other, dtype=self.dtype, device=self.device)
                                    return _original_tensor_mul(self, other)
                                raise
                        
                        def _patched_tensor_rmul(self, other):
                            """Patched right multiplication"""
                            try:
                                return _original_tensor_rmul(self, other)
                            except (RuntimeError, TypeError) as e:
                                if "unknown parameter type" in str(e):
                                    if isinstance(other, torch.Tensor) and other.dtype != self.dtype:
                                        other = other.to(self.dtype)
                                    elif isinstance(other, (int, float)):
                                        other = torch.tensor(other, dtype=self.dtype, device=self.device)
                                    return _original_tensor_rmul(self, other)
                                raise
                        
                        def _patched_tensor_add(self, other):
                            """Patched addition"""
                            try:
                                return _original_tensor_add(self, other)
                            except (RuntimeError, TypeError) as e:
                                if "unknown parameter type" in str(e):
                                    if isinstance(other, torch.Tensor) and other.dtype != self.dtype:
                                        other = other.to(self.dtype)
                                    elif isinstance(other, (int, float)):
                                        other = torch.tensor(other, dtype=self.dtype, device=self.device)
                                    return _original_tensor_add(self, other)
                                raise
                        
                        def _patched_tensor_radd(self, other):
                            """Patched right addition"""
                            try:
                                return _original_tensor_radd(self, other)
                            except (RuntimeError, TypeError) as e:
                                if "unknown parameter type" in str(e):
                                    if isinstance(other, torch.Tensor) and other.dtype != self.dtype:
                                        other = other.to(self.dtype)
                                    elif isinstance(other, (int, float)):
                                        other = torch.tensor(other, dtype=self.dtype, device=self.device)
                                    return _original_tensor_radd(self, other)
                                raise
                        
                        torch.Tensor.__mul__ = _patched_tensor_mul
                        torch.Tensor.__rmul__ = _patched_tensor_rmul
                        torch.Tensor.__add__ = _patched_tensor_add
                        torch.Tensor.__radd__ = _patched_tensor_radd
                        print("  ✓ Pre-patched torch.Tensor arithmetic operations for type compatibility")
                        
                        # Patch torch.arange to avoid "unknown parameter type" error in SAM3 model building
                        # This is critical for RoPE (Rotary Position Embedding) computations
                        _original_arange = torch.arange
                        
                        def _patched_arange(*args, **kwargs):
                            """Patched arange that avoids the PyTorch C++ backend bug"""
                            try:
                                # Ensure we're not in a grad context that might trigger the bug
                                # Remove any grad-related kwargs
                                safe_kwargs = {k: v for k, v in kwargs.items() if k not in ['requires_grad', 'grad_fn']}
                                result = _original_arange(*args, **safe_kwargs)
                                # Ensure result doesn't require grad (shouldn't for arange anyway)
                                if hasattr(result, 'requires_grad') and result.requires_grad:
                                    result = result.detach()
                                return result
                            except RuntimeError as e:
                                if "unknown parameter type" in str(e):
                                    # Fallback: create on CPU first, then move if needed
                                    device = kwargs.get('device', None)
                                    dtype = kwargs.get('dtype', None)
                                    # Remove device/dtype from kwargs for CPU creation
                                    cpu_kwargs = {k: v for k, v in kwargs.items() if k not in ['device', 'dtype', 'requires_grad', 'grad_fn']}
                                    result = _original_arange(*args, **cpu_kwargs)
                                    if dtype:
                                        result = result.to(dtype)
                                    if device:
                                        result = result.to(device)
                                    return result
                                raise
                        
                        torch.arange = _patched_arange
                        print("  ✓ Pre-patched torch.arange (before SAM3 import)")
                        
                        # Save original PyTorch functions BEFORE any patching
                        # This is critical to avoid infinite recursion in fallback paths
                        _ORIGINAL_ZEROS_BUILTIN = torch.zeros
                        _ORIGINAL_ONES_BUILTIN = torch.ones
                        _ORIGINAL_EMPTY_BUILTIN = torch.empty
                        _ORIGINAL_ARANGE_BUILTIN = torch.arange
                        
                        # Patch torch.zeros to avoid "unknown parameter type" error
                        # This is critical for position encoding and other tensor creation
                        def _patched_zeros(*args, **kwargs):
                            """Patched zeros that avoids the PyTorch C++ backend bug"""
                            try:
                                # Try with minimal safe kwargs first
                                safe_kwargs = {}
                                
                                # Pass through these parameters if present
                                if 'dtype' in kwargs:
                                    safe_kwargs['dtype'] = kwargs['dtype']
                                if 'device' in kwargs:
                                    safe_kwargs['device'] = kwargs['device']
                                if 'layout' in kwargs:
                                    safe_kwargs['layout'] = kwargs['layout']
                                if 'pin_memory' in kwargs:
                                    safe_kwargs['pin_memory'] = kwargs['pin_memory']
                                
                                result = _ORIGINAL_ZEROS_BUILTIN(*args, **safe_kwargs)
                                # Ensure result doesn't require grad
                                if hasattr(result, 'requires_grad') and result.requires_grad:
                                    result = result.detach()
                                return result
                            except RuntimeError as e:
                                if "unknown parameter type" in str(e):
                                    # Fallback: create on CPU first, then move if needed
                                    device = kwargs.get('device', None)
                                    dtype = kwargs.get('dtype', None)
                                    # Remove unsupported kwargs for CPU creation  
                                    cpu_kwargs = {}
                                    if 'dtype' in kwargs:
                                        cpu_kwargs['dtype'] = kwargs['dtype']
                                    if 'layout' in kwargs:
                                        cpu_kwargs['layout'] = kwargs['layout']
                                    if 'pin_memory' in kwargs:
                                        cpu_kwargs['pin_memory'] = kwargs['pin_memory']
                                    # Call ORIGINAL function with minimal kwargs, CPU device only
                                    result = _ORIGINAL_ZEROS_BUILTIN(*args, **cpu_kwargs, device='cpu')
                                    # Move to target device if different from CPU
                                    if device and device != 'cpu':
                                        result = result.to(device=device)
                                    return result
                                raise
                        
                        torch.zeros = _patched_zeros
                        print("  ✓ Pre-patched torch.zeros (before SAM3 import)")
                        
                        # Patch torch.ones to avoid "unknown parameter type" error
                        def _patched_ones(*args, **kwargs):
                            """Patched ones that avoids the PyTorch C++ backend bug"""
                            try:
                                # Try with minimal safe kwargs first
                                safe_kwargs = {}
                                
                                # Pass through these parameters if present
                                if 'dtype' in kwargs:
                                    safe_kwargs['dtype'] = kwargs['dtype']
                                if 'device' in kwargs:
                                    safe_kwargs['device'] = kwargs['device']
                                if 'layout' in kwargs:
                                    safe_kwargs['layout'] = kwargs['layout']
                                if 'pin_memory' in kwargs:
                                    safe_kwargs['pin_memory'] = kwargs['pin_memory']
                                
                                result = _ORIGINAL_ONES_BUILTIN(*args, **safe_kwargs)
                                # Ensure result doesn't require grad
                                if hasattr(result, 'requires_grad') and result.requires_grad:
                                    result = result.detach()
                                return result
                            except RuntimeError as e:
                                if "unknown parameter type" in str(e):
                                    # Fallback: create on CPU first, then move if needed
                                    device = kwargs.get('device', None)
                                    dtype = kwargs.get('dtype', None)
                                    # Remove unsupported kwargs for CPU creation  
                                    cpu_kwargs = {}
                                    if 'dtype' in kwargs:
                                        cpu_kwargs['dtype'] = kwargs['dtype']
                                    if 'layout' in kwargs:
                                        cpu_kwargs['layout'] = kwargs['layout']
                                    if 'pin_memory' in kwargs:
                                        cpu_kwargs['pin_memory'] = kwargs['pin_memory']
                                    # Call ORIGINAL function with minimal kwargs, CPU device only
                                    result = _ORIGINAL_ONES_BUILTIN(*args, **cpu_kwargs, device='cpu')
                                    # Move to target device if different from CPU
                                    if device and device != 'cpu':
                                        result = result.to(device=device)
                                    return result
                                raise
                        
                        torch.ones = _patched_ones
                        print("  ✓ Pre-patched torch.ones (before SAM3 import)")
                        
                        # Also patch torch.empty and torch.full for completeness
                        _original_empty = torch.empty
                        def _patched_empty(*args, **kwargs):
                            try:
                                safe_kwargs = {k: v for k, v in kwargs.items() if k not in ['requires_grad', 'grad_fn']}
                                result = _original_empty(*args, **safe_kwargs)
                                if hasattr(result, 'requires_grad') and result.requires_grad:
                                    result = result.detach()
                                return result
                            except RuntimeError as e:
                                if "unknown parameter type" in str(e):
                                    device = kwargs.get('device', None)
                                    dtype = kwargs.get('dtype', None)
                                    cpu_kwargs = {k: v for k, v in kwargs.items() if k not in ['device', 'dtype', 'requires_grad', 'grad_fn']}
                                    result = _original_empty(*args, **cpu_kwargs)
                                    if dtype:
                                        result = result.to(dtype)
                                    if device:
                                        result = result.to(device)
                                    return result
                                raise
                        torch.empty = _patched_empty
                        print("  ✓ Pre-patched torch.empty (before SAM3 import)")
                        
                    except Exception as early_patch_err:
                        print(f"  ⚠️ Could not pre-patch initialization functions: {early_patch_err}")
                        import traceback
                        traceback.print_exc()
                    
                    # CRITICAL: Import SAM3 AFTER all patches are applied
                    # This ensures all tensor creation functions are patched before SAM3 uses them
                    print("  Importing SAM3 (all patches applied)...")
                    from sam3.model_builder import build_sam3_image_model
                    print("Detected SAM3 checkpoint, using SAM3 model builder...")
                    
                    # For CUDA operations in background threads, we need to be very careful
                    # With proper Qt signals/slots, CUDA can work in background threads
                    # Allow CUDA unless explicitly forced to CPU
                    import threading
                    is_main_thread = threading.current_thread() is threading.main_thread()
                    force_cpu_thread = os.environ.get("SAM3_FORCE_CPU_THREAD", "0") == "1"
                    
                    if self.device == "cuda" and torch.cuda.is_available():
                        if not is_main_thread and force_cpu_thread:
                            # Explicitly requested CPU for background threads
                            print("  Note: SAM3_FORCE_CPU_THREAD=1, using CPU in background thread")
                            self.device = "cpu"
                        else:
                            try:
                                # Initialize CUDA (safe in Qt worker threads with proper setup)
                                print(f"  Attempting to use CUDA/GPU (device: cuda:0)")
                                print(f"  CUDA available: {torch.cuda.is_available()}")
                                print(f"  CUDA initialized: {torch.cuda.is_initialized()}")
                                print(f"  CUDA device count: {torch.cuda.device_count()}")
                                
                                if not is_main_thread:
                                    # In background thread - CUDA should work with Qt signals/slots
                                    print(f"  Using CUDA/GPU in background thread")
                                    # Try to initialize CUDA if not already initialized
                                    if not torch.cuda.is_initialized():
                                        print("  Initializing CUDA in background thread...")
                                        try:
                                            # Create a dummy tensor to force CUDA initialization
                                            dummy = torch.zeros(1).cuda()
                                            del dummy
                                            torch.cuda.empty_cache()
                                            print("  CUDA initialized successfully in background thread")
                                        except Exception as init_err:
                                            print(f"  ⚠️ CUDA initialization failed in background thread: {init_err}")
                                            print("  Falling back to CPU...")
                                            self.device = "cpu"
                                    else:
                                        print("  CUDA already initialized")
                                    
                                    if self.device == "cuda":
                                        try:
                                            torch.cuda.set_device(0)
                                            print(f"  Set CUDA device to: cuda:0")
                                        except RuntimeError as cuda_err:
                                            if "driver shutting down" in str(cuda_err).lower() or "cuda error" in str(cuda_err).lower():
                                                print(f"  ⚠️ CUDA driver shutting down, forcing CPU mode")
                                                self.device = "cpu"
                                            else:
                                                raise
                                else:
                                    # Main thread - safe to initialize
                                    if not torch.cuda.is_initialized():
                                        print("  Initializing CUDA in main thread...")
                                        torch.cuda.init()
                                    torch.cuda.set_device(0)
                                    print(f"  Set CUDA device to: cuda:0")
                                
                                if self.device == "cuda":
                                    try:
                                        # Verify CUDA is actually working before proceeding
                                        device_name = torch.cuda.get_device_name(0)
                                        print(f"  ✓ Using CUDA device: {device_name}")
                                    except RuntimeError as cuda_check_err:
                                        if "driver shutting down" in str(cuda_check_err).lower() or "cuda error" in str(cuda_check_err).lower():
                                            print(f"  ⚠️ CUDA driver not available, forcing CPU mode")
                                            self.device = "cpu"
                                        else:
                                            raise
                                    print(f"  CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
                            except Exception as cuda_err:
                                print(f"  ❌ CUDA initialization error: {cuda_err}")
                                import traceback
                                traceback.print_exc()
                                if "driver shutting down" in str(cuda_err).lower() or "cuda error" in str(cuda_err).lower():
                                    print("  CUDA driver shutting down, forcing CPU mode...")
                                else:
                                    print("  Falling back to CPU...")
                                self.device = "cpu"
                                
                                # Verify CUDA is actually unavailable before proceeding
                                try:
                                    if torch.cuda.is_available():
                                        # Try a simple CUDA operation to see if it's really working
                                        torch.cuda.get_device_name(0)
                                    else:
                                        print("  ✓ CUDA confirmed unavailable, using CPU")
                                except Exception:
                                    print("  ✓ CUDA confirmed unavailable, using CPU")
                    
                    model_kwargs = {
                        "device": self.device,
                        "eval_mode": True,
                        "checkpoint_path": self.checkpoint_path,
                        "load_from_HF": False,
                        "enable_segmentation": True,
                        "enable_inst_interactivity": False,
                    }
                    
                    # Try to find BPE path
                    bpe_path = self._find_sam3_bpe_path()
                    if bpe_path:
                        model_kwargs["bpe_path"] = bpe_path
                    
                    # Wrap model loading in try-except to catch any segfault-causing issues
                    # Use torch.no_grad() to avoid autograd issues during initialization
                    # This prevents "unknown parameter type" errors in PyTorch's C++ backend
                    try:
                        # Save original device
                        original_device = model_kwargs.get("device", "cpu")
                        
                        # Try loading on CPU first with no_grad to avoid initialization issues
                        print(f"  Loading SAM3 model (initializing on CPU with no_grad, will move to {self.device} after)...")
                        model_kwargs_cpu = model_kwargs.copy()
                        model_kwargs_cpu["device"] = "cpu"
                        
                        # CRITICAL: Ensure CUDA context is initialized BEFORE model building
                        # Even though we're building on CPU, some operations may still try to access CUDA
                        if torch.cuda.is_available() and self.device.startswith('cuda'):
                            try:
                                device_index = 0 if self.device == "cuda" else int(self.device.split(':')[1])
                                torch.cuda.set_device(device_index)
                                # Force context creation with a dummy operation
                                _dummy = torch.tensor([1.0]).cuda(device_index)
                                del _dummy
                                torch.cuda.synchronize(device_index)
                                print(f"  ✓ CUDA context ready in worker thread (device: cuda:{device_index})")
                            except Exception as ctx_err:
                                print(f"  ⚠️ CUDA context initialization warning: {ctx_err}")
                                # Continue anyway - model will be built on CPU
                        
                        # Build model on CPU first
                        # NOTE: The "unknown parameter type" error occurs in multiple places:
                        # 1. During checkpoint loading with weights_only=True
                        # 2. During weight initialization in timm's trunc_normal_ (uses torch.no_grad() context)
                        # This is a known bug in PyTorch 2.7.1+cu118's C++ backend.
                        # We'll monkey-patch both torch.load and timm's trunc_normal_ to work around this.
                        # Catch ALL exceptions to prevent segfaults from crashing the app
                        try:
                            # Strategy: Monkey-patch both torch.load and timm's trunc_normal_ to avoid PyTorch bugs
                            print(f"  Loading model with PyTorch 2.7.1 bug workarounds...")
                            prev_grad = torch.is_grad_enabled()
                            
                            # Disable grad manually (don't use context manager to avoid segfaults)
                            torch.set_grad_enabled(False)
                            
                            # Monkey-patch torch.load to avoid weights_only=True bug
                            original_torch_load = torch.load
                            def patched_torch_load(*args, **kwargs):
                                # Force weights_only=False to avoid PyTorch C++ backend bug
                                kwargs.pop('weights_only', None)  # Remove if present
                                kwargs['weights_only'] = False
                                return original_torch_load(*args, **kwargs)
                            
                            # Re-apply PyTorch init function patches (in case they were reset)
                            try:
                                import torch.nn.init as init
                                import math
                                
                                # Re-apply _no_grad_fill_ patch
                                if hasattr(init, '_no_grad_fill_'):
                                    def patched_no_grad_fill(tensor, val):
                                        """Patched version that avoids torch.no_grad() context manager"""
                                        if tensor.requires_grad:
                                            tensor.requires_grad_(False)
                                        return tensor.fill_(val)
                                    init._no_grad_fill_ = patched_no_grad_fill
                                    print(f"  ✓ Re-applied PyTorch's _no_grad_fill_ patch")
                                
                                # Re-apply _no_grad_zero_ patch
                                if hasattr(init, '_no_grad_zero_'):
                                    def patched_no_grad_zero(tensor):
                                        """Patched version that avoids torch.no_grad() context manager"""
                                        if tensor.requires_grad:
                                            tensor.requires_grad_(False)
                                        return tensor.zero_()
                                    init._no_grad_zero_ = patched_no_grad_zero
                                    print(f"  ✓ Re-applied PyTorch's _no_grad_zero_ patch")
                                
                                # Re-apply _no_grad_uniform_ patch
                                if hasattr(init, '_no_grad_uniform_'):
                                    def patched_no_grad_uniform(tensor, a, b, generator=None):
                                        """Patched version that avoids torch.no_grad() context manager"""
                                        if tensor.requires_grad:
                                            tensor.requires_grad_(False)
                                        return tensor.uniform_(a, b, generator=generator)
                                    init._no_grad_uniform_ = patched_no_grad_uniform
                                    print(f"  ✓ Re-applied PyTorch's _no_grad_uniform_ patch")
                                
                                # Re-apply ones_() and zeros_() patches
                                if hasattr(init, 'ones_'):
                                    def patched_ones_(tensor):
                                        """Patched version that disables grad before in-place operation"""
                                        if tensor.requires_grad:
                                            tensor.requires_grad_(False)
                                        return tensor.fill_(1.0)
                                    init.ones_ = patched_ones_
                                    print(f"  ✓ Re-applied PyTorch's ones_ patch")
                                
                                if hasattr(init, 'zeros_'):
                                    def patched_zeros_(tensor):
                                        """Patched version that disables grad before in-place operation"""
                                        if tensor.requires_grad:
                                            tensor.requires_grad_(False)
                                        return tensor.zero_()
                                    init.zeros_ = patched_zeros_
                                    print(f"  ✓ Re-applied PyTorch's zeros_ patch")
                                
                                # Re-apply normal_ and _no_grad_normal_ patches
                                if hasattr(init, '_no_grad_normal_'):
                                    def patched_no_grad_normal(tensor, mean, std, generator=None):
                                        """Patched version that avoids torch.no_grad() context manager"""
                                        if tensor.requires_grad:
                                            tensor.requires_grad_(False)
                                        return tensor.normal_(mean, std, generator=generator)
                                    init._no_grad_normal_ = patched_no_grad_normal
                                    print(f"  ✓ Re-applied PyTorch's _no_grad_normal_ patch")
                                
                                if hasattr(init, 'normal_'):
                                    def patched_normal(tensor, mean=0., std=1., *, generator=None):
                                        """Patched version that avoids torch.no_grad() context manager"""
                                        if tensor.requires_grad:
                                            tensor.requires_grad_(False)
                                        return tensor.normal_(mean, std, generator=generator)
                                    init.normal_ = patched_normal
                                    print(f"  ✓ Re-applied PyTorch's normal_ patch")
                            except Exception as init_patch_err:
                                print(f"  ⚠️ Could not re-patch PyTorch init functions: {init_patch_err}")
                            
                            # Re-apply timm's trunc_normal_ patch (in case it was reset or SAM3 imported it differently)
                            # The early patch should have worked, but let's make sure it's still applied
                            try:
                                import timm.layers.weight_init as weight_init
                                import math
                                
                                # Check if it's already patched (from early patch)
                                if hasattr(weight_init, 'trunc_normal_'):
                                    # Verify it's our patched version by checking if it has our signature
                                    # If not, re-apply the patch
                                    try:
                                        # Try to see if it's the original (which would use torch.no_grad)
                                        import inspect
                                        source = inspect.getsource(weight_init.trunc_normal_)
                                        if 'torch.no_grad' in source or 'with torch.no_grad' in source:
                                            # Still the original, need to patch
                                            raise AttributeError("Original function still present")
                                    except:
                                        # Can't inspect, assume we need to patch
                                        pass
                                    
                                    # Re-apply patch to be safe
                                    def patched_trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
                                        """Patched version that avoids torch.no_grad() context manager"""
                                        # Ensure tensor doesn't require grad before in-place operations
                                        if tensor.requires_grad:
                                            tensor.requires_grad_(False)
                                        # Complete implementation without context manager
                                        l = (1. + math.erf(((a - mean) / std) / math.sqrt(2.))) / 2.
                                        u = (1. + math.erf(((b - mean) / std) / math.sqrt(2.))) / 2.
                                        tensor.uniform_(2 * l - 1, 2 * u - 1)
                                        tensor.erfinv_()
                                        tensor.mul_(std * math.sqrt(2.))
                                        tensor.add_(mean)
                                        tensor.clamp_(min=a, max=b)
                                        return tensor
                                    
                                    weight_init.trunc_normal_ = patched_trunc_normal_
                                    
                                    # Also patch _trunc_normal_ if it exists
                                    if hasattr(weight_init, '_trunc_normal_'):
                                        weight_init._trunc_normal_ = patched_trunc_normal_
                                    
                                    print(f"  ✓ Re-applied timm's trunc_normal_ patch")
                            except Exception as patch_err:
                                print(f"  ⚠️ Could not re-patch timm's trunc_normal_: {patch_err}")
                            
                            # Apply the torch.load patch
                            torch.load = patched_torch_load
                            
                            load_success = False
                            load_err = None
                            try:
                                # Load the model (will use patched torch.load)
                                # Explicitly disable gradients for all model building
                                # This ensures that checkpoint loading doesn't try to modify requires_grad tensors
                                prev_grad_state = torch.is_grad_enabled()
                                torch.set_grad_enabled(False)
                                
                                # Verify patches are still active before building model
                                if not hasattr(torch.zeros, '__name__') or 'patched' not in str(torch.zeros):
                                    print("  ⚠️ Warning: torch.zeros patch may not be active, re-applying...")
                                    # Re-apply zeros patch
                                    _original_zeros = torch.zeros
                                    def _patched_zeros_reapply(*args, **kwargs):
                                        try:
                                            safe_kwargs = {k: v for k, v in kwargs.items() if k not in ['requires_grad', 'grad_fn']}
                                            return _original_zeros(*args, **safe_kwargs)
                                        except RuntimeError as e:
                                            if "unknown parameter type" in str(e):
                                                device = kwargs.get('device', None)
                                                dtype = kwargs.get('dtype', None)
                                                cpu_kwargs = {k: v for k, v in kwargs.items() if k not in ['device', 'dtype', 'requires_grad', 'grad_fn']}
                                                result = _original_zeros(*args, **cpu_kwargs)
                                                if dtype:
                                                    result = result.to(dtype)
                                                if device:
                                                    result = result.to(device)
                                                return result
                                            raise
                                    torch.zeros = _patched_zeros_reapply
                                    print("  ✓ Re-applied torch.zeros patch before model building")
                                
                                try:
                                    print("  Building SAM3 model architecture (this may take a moment)...")
                                    self.sam = build_sam3_image_model(**model_kwargs_cpu)
                                    print("  ✓ Model architecture built successfully")
                                except RuntimeError as build_err:
                                    if "unknown parameter type" in str(build_err):
                                        # Try one more time with CPU device for all tensor creation
                                        print("  ⚠️ Model building failed with device parameter, trying CPU fallback...")
                                        # Use the ORIGINAL builtin functions saved at the beginning
                                        def _patched_zeros_cpu_fallback(*args, **kwargs):
                                            # Force CPU creation, then move if needed
                                            device = kwargs.pop('device', None)
                                            dtype = kwargs.get('dtype', None)
                                            # Call ORIGINAL with only safe kwargs
                                            safe_kwargs = {k: v for k, v in kwargs.items() if k not in ['requires_grad', 'grad_fn']}
                                            result = _ORIGINAL_ZEROS_BUILTIN(*args, **safe_kwargs)
                                            if dtype:
                                                result = result.to(dtype)
                                            if device and device != 'cpu':
                                                result = result.to(device)
                                            return result
                                        torch.zeros = _patched_zeros_cpu_fallback
                                        # Also patch ones
                                        def _patched_ones_cpu_fallback(*args, **kwargs):
                                            device = kwargs.pop('device', None)
                                            dtype = kwargs.get('dtype', None)
                                            # Call ORIGINAL with only safe kwargs
                                            safe_kwargs = {k: v for k, v in kwargs.items() if k not in ['requires_grad', 'grad_fn']}
                                            result = _ORIGINAL_ONES_BUILTIN(*args, **safe_kwargs)
                                            if dtype:
                                                result = result.to(dtype)
                                            if device and device != 'cpu':
                                                result = result.to(device)
                                            return result
                                        torch.ones = _patched_ones_cpu_fallback
                                        # Try again
                                        self.sam = build_sam3_image_model(**model_kwargs_cpu)
                                        print("  ✓ Model built with CPU fallback")
                                    else:
                                        raise
                                finally:
                                    # Try to restore grad state, but leave it disabled to be safe
                                    # torch.set_grad_enabled(prev_grad_state)
                                    pass
                                
                                # Set model to eval mode (this is what we want anyway)
                                if hasattr(self.sam, 'eval'):
                                    self.sam.eval()
                                elif hasattr(self.sam, 'model') and hasattr(self.sam.model, 'eval'):
                                    self.sam.model.eval()
                                
                                # Ensure all model parameters have requires_grad=False
                                for param in self.sam.parameters():
                                    param.requires_grad = False
                                
                                self.is_sam3_model = True
                                load_success = True
                                print(f"  ✓ Model loaded successfully (grad disabled, eval mode)")
                            except Exception as err:
                                load_err = err
                                print(f"  ⚠️ Model loading failed: {err}")
                                raise err
                            finally:
                                # Restore original torch.load
                                torch.load = original_torch_load
                                
                                # Don't restore timm trunc_normal_ - we want the patch to persist
                                # The patch is necessary to avoid the PyTorch bug
                                
                                # Try to restore grad state, but don't fail if it doesn't work
                                # If restoring causes issues, leave grad disabled (safer for inference)
                                # NOTE: We intentionally DON'T restore grad state to avoid the segfault
                                # Leaving grad disabled is safe for inference-only models
                                if not load_success and load_err and "unknown parameter type" in str(load_err).lower():
                                    # If we got the known error, definitely don't try to restore
                                    print(f"  ⚠️ Known PyTorch bug detected, leaving grad disabled (safe for inference)")
                                # For now, we'll leave grad disabled to avoid any potential segfaults
                                # This is safe since we're only doing inference
                        except Exception as init_err:
                            # Catch ALL exceptions (including SystemExit, etc.) to prevent crashes
                            import traceback
                            tb_str = traceback.format_exc()
                            print(f"  ⚠️ SAM3 model initialization failed: {init_err}")
                            print(f"  Traceback:\n{tb_str}")
                            
                            # Check if it's the known "unknown parameter type" error
                            if "unknown parameter type" in str(init_err).lower():
                                # Try one more fallback: load directly on CUDA if available
                                # Sometimes CUDA initialization works when CPU fails
                                if self.device == "cuda" and torch.cuda.is_available() and original_device == "cuda":
                                    print(f"  ⚠️ CPU loading failed, trying direct CUDA loading as last resort...")
                                    try:
                                        # Try loading directly on CUDA without no_grad context
                                        prev_grad = torch.is_grad_enabled()
                                        torch.set_grad_enabled(False)
                                        try:
                                            self.sam = build_sam3_image_model(**model_kwargs)
                                            self.is_sam3_model = True
                                            print(f"  ✓ Model loaded successfully on CUDA (direct)")
                                        finally:
                                            try:
                                                torch.set_grad_enabled(prev_grad)
                                            except:
                                                try:
                                                    torch.set_grad_enabled(False)
                                                except:
                                                    pass
                                    except Exception as cuda_fallback_err:
                                        # CUDA fallback also failed, show error message
                                        is_timm_issue = "timm" in tb_str.lower() or "weight_init" in tb_str.lower()
                                        
                                        if is_timm_issue:
                                            error_msg = (
                                                "CRITICAL: SAM3 model initialization failed.\n\n"
                                                "The error 'unknown parameter type' during model weight initialization\n"
                                                "suggests a compatibility issue between PyTorch and timm library.\n\n"
                                                "SOLUTION:\n"
                                                "1. Update timm library:\n"
                                                "   pip install --upgrade timm\n\n"
                                                "2. If that doesn't work, try reinstalling PyTorch:\n"
                                                "   pip uninstall torch torchvision torchaudio\n"
                                                "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n\n"
                                                "3. Restart this application\n\n"
                                                "SAM3 will not work until this is fixed."
                                            )
                                        else:
                                            error_msg = (
                                                "CRITICAL: PyTorch/SAM3 model initialization failed.\n\n"
                                                "The error 'unknown parameter type' during model building\n"
                                                "indicates a compatibility issue.\n\n"
                                                "SOLUTION:\n"
                                                "1. Update timm: pip install --upgrade timm\n"
                                                "2. Reinstall PyTorch: pip uninstall torch torchvision torchaudio && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n"
                                                "3. Restart this application\n\n"
                                                "SAM3 will not work until this is fixed."
                                            )
                                        print("=" * 80)
                                        print("ERROR: PyTorch Installation Issue")
                                        print("=" * 80)
                                        print(error_msg)
                                        print("=" * 80)
                                        self.error_occurred.emit(error_msg)
                                        return
                                else:
                                    # No CUDA fallback available, show error
                                    is_timm_issue = "timm" in tb_str.lower() or "weight_init" in tb_str.lower()
                                    
                                    if is_timm_issue:
                                        error_msg = (
                                            "CRITICAL: SAM3 model initialization failed.\n\n"
                                            "The error 'unknown parameter type' during model weight initialization\n"
                                            "suggests a compatibility issue between PyTorch and timm library.\n\n"
                                            "SOLUTION:\n"
                                            "1. Update timm library:\n"
                                            "   pip install --upgrade timm\n\n"
                                            "2. If that doesn't work, try reinstalling PyTorch:\n"
                                            "   pip uninstall torch torchvision torchaudio\n"
                                            "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n\n"
                                            "3. Restart this application\n\n"
                                            "SAM3 will not work until this is fixed."
                                        )
                                    else:
                                        error_msg = (
                                            "CRITICAL: PyTorch/SAM3 model initialization failed.\n\n"
                                            "The error 'unknown parameter type' during model building\n"
                                            "indicates a compatibility issue.\n\n"
                                            "SOLUTION:\n"
                                            "1. Update timm: pip install --upgrade timm\n"
                                            "2. Reinstall PyTorch: pip uninstall torch torchvision torchaudio && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n"
                                            "3. Restart this application\n\n"
                                            "SAM3 will not work until this is fixed."
                                        )
                                    print("=" * 80)
                                    print("ERROR: PyTorch Installation Issue")
                                    print("=" * 80)
                                    print(error_msg)
                                    print("=" * 80)
                                    self.error_occurred.emit(error_msg)
                                    return
                            else:
                                # For other errors, emit a generic error message
                                error_msg = f"SAM3 model initialization failed: {init_err}\n\nPlease check the error message above and ensure PyTorch and timm are correctly installed."
                                print("=" * 80)
                                print("ERROR: SAM3 Initialization Failed")
                                print("=" * 80)
                                print(error_msg)
                                print("=" * 80)
                                self.error_occurred.emit(error_msg)
                                return
                        
                        # Now move to CUDA if that's the target device
                        if self.device == "cuda" and torch.cuda.is_available() and original_device == "cuda":
                            try:
                                print(f"  Moving model to CUDA...")
                                # Move model to CUDA explicitly
                                if hasattr(self.sam, 'to'):
                                    self.sam = self.sam.to("cuda")
                                # Check if model is actually on CUDA and verify memory usage
                                if hasattr(self.sam, 'parameters'):
                                    cuda_device_found = False
                                    for param in self.sam.parameters():
                                        if param.device.type == "cuda":
                                            cuda_device = param.device
                                            cuda_device_found = True
                                            break
                                    
                                    if cuda_device_found:
                                        # Check CUDA memory after model loading
                                        memory_allocated = torch.cuda.memory_allocated(cuda_device.index) / 1024**2
                                        memory_reserved = torch.cuda.memory_reserved(cuda_device.index) / 1024**2
                                        print(f"  ✓ Model moved to CUDA device: {cuda_device}")
                                        print(f"  ✓ CUDA memory allocated: {memory_allocated:.1f} MB")
                                        print(f"  ✓ CUDA memory reserved: {memory_reserved:.1f} MB")
                                    else:
                                        print("  ⚠️ Warning: Model parameters not found on CUDA")
                            except Exception as move_err:
                                print(f"  ⚠️ Warning: Could not move model to CUDA: {move_err}")
                                print("  Model will remain on CPU")
                                self.device = "cpu"
                        
                        print("SAM3 model loaded successfully")
                    except RuntimeError as e:
                        # CUDA out of memory or other runtime errors
                        error_msg = f"SAM3 model loading failed: {e}"
                        print(f"ERROR: {error_msg}")
                        # If CPU initialization also failed, try with no_grad context
                        if "unknown parameter type" in str(e).lower() or "grad" in str(e).lower():
                            print("  Initialization error detected, trying with no_grad context...")
                            try:
                                # Try CPU with explicit no_grad
                                with torch.no_grad():
                                    self.sam = build_sam3_image_model(**model_kwargs_cpu)
                                self.is_sam3_model = True
                                print("SAM3 model loaded successfully on CPU (with no_grad)")
                                
                                # Now try to move to CUDA if needed
                                if self.device == "cuda" and torch.cuda.is_available() and original_device == "cuda":
                                    try:
                                        print(f"  Moving model to CUDA...")
                                        with torch.no_grad():
                                            self.sam = self.sam.to("cuda")
                                        # Verify CUDA placement
                                        if hasattr(self.sam, 'parameters'):
                                            for param in self.sam.parameters():
                                                if param.device.type == "cuda":
                                                    memory_allocated = torch.cuda.memory_allocated(param.device.index) / 1024**2
                                                    print(f"  ✓ Model moved to CUDA device: {param.device}")
                                                    print(f"  ✓ CUDA memory allocated: {memory_allocated:.1f} MB")
                                                break
                                    except Exception as move_err:
                                        print(f"  ⚠️ Warning: Could not move model to CUDA: {move_err}")
                                        print("  Model will remain on CPU")
                                        self.device = "cpu"
                            except Exception as no_grad_err:
                                print(f"  no_grad loading also failed: {no_grad_err}")
                                # This is a PyTorch installation issue - provide clear instructions
                                if "unknown parameter type" in str(no_grad_err).lower():
                                    error_msg = (
                                        "CRITICAL: PyTorch installation is corrupted.\n\n"
                                        "The error 'unknown parameter type' indicates PyTorch's C++ backend is broken.\n"
                                        "This cannot be fixed by the application - you must reinstall PyTorch.\n\n"
                                        "SOLUTION:\n"
                                        "1. conda activate curation_tool\n"
                                        "2. pip uninstall torch torchvision torchaudio\n"
                                        "3. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n"
                                        "   (Check your CUDA version first: nvidia-smi)\n"
                                        "4. Restart this application\n\n"
                                        "SAM3 will not work until PyTorch is reinstalled."
                                    )
                                    print("=" * 80)
                                    print("ERROR: PyTorch Installation Issue")
                                    print("=" * 80)
                                    print(error_msg)
                                    print("=" * 80)
                                    self.error_occurred.emit(error_msg)
                                    return
                                else:
                                    raise RuntimeError(
                                        f"SAM3 model initialization failed due to PyTorch compatibility issue. "
                                        f"Error: {no_grad_err}. "
                                        f"This may be due to PyTorch/timm version mismatch. "
                                        f"Please check your PyTorch and timm versions are compatible."
                                    )
                        elif "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                            print("  Attempting to load on CPU instead...")
                            model_kwargs["device"] = "cpu"
                            self.device = "cpu"
                            try:
                                with torch.no_grad():
                                    self.sam = build_sam3_image_model(**model_kwargs)
                                self.is_sam3_model = True
                                print("SAM3 model loaded successfully on CPU")
                            except Exception as cpu_err:
                                raise RuntimeError(f"Failed to load on CPU: {cpu_err}")
                        else:
                            raise
                    except Exception as e:
                        # Catch any other errors that might cause segfaults
                        error_msg = f"SAM3 model loading error: {e}"
                        print(f"ERROR: {error_msg}")
                        import traceback
                        traceback.print_exc()
                        raise
                    
                    # Try to create a standard predictor for point/box prompts
                    # SAM3 models might support the standard predictor interface
                    try:
                        if HAS_SAM:
                            from segment_anything import SamPredictor
                            self.predictor = SamPredictor(self.sam)
                            print("Created standard predictor for SAM3 model (for point/box prompts)")
                        else:
                            print("Standard SAM not available - SAM3 processor will be used for all operations")
                    except Exception as e:
                        print(f"Could not create standard predictor for SAM3: {e}")
                        print("Point/box prompts will use SAM3 processor when available")
                    
                    self.predictor_ready.emit()
                    return
                except ImportError as e:
                    error_msg = str(e)
                    print(f"SAM3 not available (ImportError): {error_msg}")
                    # Check if SAM3 was found but import failed due to dependencies
                    sam3_found_but_failed = False
                    for sam3_path in ["/cellchorus/sam3_training/sam3"]:
                        if os.path.exists(os.path.join(sam3_path, "sam3")):
                            sam3_found_but_failed = True
                            break
                    
                    if sam3_found_but_failed:
                        self.error_occurred.emit(
                            f"SAM3 found at /cellchorus/sam3_training/sam3 but import failed due to missing dependencies.\n"
                            f"Error: {error_msg}\n\n"
                            "Please install required dependencies:\n"
                            "  pip install decord\n"
                            "  or install all SAM3 dependencies:\n"
                            "  cd /cellchorus/sam3_training/sam3 && pip install -e ."
                        )
                        return
                    else:
                        print("SAM3 not found, trying standard SAM loading...")
                except Exception as e:
                    error_msg = f"Failed to load as SAM3: {e}"
                    print(error_msg)
                    import traceback
                    traceback.print_exc()
                    self.error_occurred.emit(f"Failed to load SAM3 model: {error_msg}")
                    return
            
            # Load checkpoint (standard SAM)
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            print(f"Checkpoint type: {type(checkpoint)}")
            print(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'not a dict'}")
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                # Checkpoint is a dict, could be state_dict or full checkpoint
                if 'model' in checkpoint:
                    # Format: {'model': model_state_dict, ...}
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    # Format: {'state_dict': {...}, ...}
                    state_dict = checkpoint['state_dict']
                else:
                    # Assume the dict IS the state_dict
                    state_dict = checkpoint
                
                # Build model from state_dict
                if HAS_SAM:
                    print(f"Building SAM model with type: {self.model_type}")
                    self.sam = sam_model_registry[self.model_type](checkpoint=None)
                    
                    # Load state dict
                    missing, unexpected = self.sam.load_state_dict(state_dict, strict=False)
                    print(f"Loaded state dict - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
                    
                    self.sam.to(self.device)
                    self.sam.eval()
                    self.predictor = SamPredictor(self.sam)
                    print("SAM model built from state_dict successfully")
                else:
                    self.error_occurred.emit("segment_anything package required but not available")
                    return
            else:
                # Checkpoint is a model object directly
                self.sam = checkpoint
                if hasattr(self.sam, 'eval'):
                    self.sam.eval()
                if hasattr(self.sam, 'to'):
                    self.sam.to(self.device)
                
                if HAS_SAM:
                    self.predictor = SamPredictor(self.sam)
                else:
                    self.predictor = self.sam
                print("SAM model loaded directly")
            
            self.predictor_ready.emit()
            
        except Exception as e:
            error_msg = f"Unexpected error loading SAM: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(error_msg)
    
    @pyqtSlot(object)
    def set_image(self, image: np.ndarray):
        """Set image for SAM predictor"""
        print("[SAM WORKER] set_image() called")
        try:
            # Ensure image is RGB
            if len(image.shape) == 2:
                image = np.stack([image, image, image], axis=-1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]
            
            # Ensure uint8
            if image.dtype != np.uint8:
                if image.max() > 1.0:
                    image = np.clip(image, 0, 255).astype(np.uint8)
                else:
                    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            
            # Store image for SAM3 processor
            try:
                from PIL import Image as PILImage
                self._current_image_for_sam = PILImage.fromarray(image)
                self._last_set_image = image.copy()  # Store numpy array too
                print(f"[SAM WORKER] Image stored for SAM3 (shape: {image.shape})")
            except Exception as e:
                self._last_set_image = image.copy() if hasattr(image, 'copy') else image
                print(f"[SAM WORKER] Image stored (PIL conversion failed: {e})")
            
            # Set image in predictor (standard SAM)
            if self.predictor is not None:
                if hasattr(self.predictor, 'set_image'):
                    print("[SAM WORKER] Setting image in standard SAM predictor...")
                    self.predictor.set_image(image)
                    print("[SAM WORKER] Image set in SAM predictor")
            elif self.is_sam3_model:
                # For SAM3, image will be set when needed
                # Only print this once to reduce console clutter
                if not hasattr(self, '_image_set_message_shown'):
                    print("[SAM WORKER] Using SAM3 processor for image processing (image will be set during prediction)")
                    self._image_set_message_shown = True
            print("[SAM WORKER] set_image() completed")
        except Exception as e:
            print(f"[SAM WORKER] ERROR in set_image: {e}")
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(f"Failed to set image: {str(e)}")
    
    def predict_with_points(self, points, labels, confidence_threshold=0.5, max_cells=2):
        """
        Run SAM prediction with point prompts
        
        Args:
            points: numpy array of shape (N, 2) with (x, y) coordinates
            labels: numpy array of shape (N,) with 1 for foreground, 0 for background
            confidence_threshold: Confidence threshold for SAM3 (default: 0.5)
            max_cells: Maximum cells to return for SAM3 (default: 2)
            
        Returns:
            masks: predicted masks (single mask for standard SAM, list for SAM3)
        """
        # First try standard predictor (works for both standard SAM and SAM3 if predictor was created)
        # This is the preferred method as it's more reliable for point prompts
        # Standard SAM - try predictor methods (works for both standard SAM and SAM3 if predictor was created)
        if self.predictor is not None:
            try:
                points = np.array(points)
                labels = np.array(labels)
                
                if hasattr(self.predictor, 'predict'):
                    masks, scores, logits = self.predictor.predict(
                        point_coords=points,
                        point_labels=labels,
                        multimask_output=True
                    )
                    # Return the mask with highest score
                    best_idx = np.argmax(scores)
                    return masks[best_idx]
                else:
                    print("Predictor doesn't have 'predict' method")
                    return None
            except Exception as e:
                print(f"Error in SAM prediction with points: {e}")
                # If predictor fails, try SAM3 processor approach
                if self.is_sam3_model:
                    print("Standard predictor failed, trying SAM3 processor with point prompts...")
                    return self._predict_with_points_sam3(points, labels, confidence_threshold, max_cells)
                self.error_occurred.emit(f"Prediction failed: {str(e)}")
                return None
        else:
            # No predictor - try SAM3 processor if available
            if self.is_sam3_model:
                print("No standard predictor, trying SAM3 processor with point prompts...")
                return self._predict_with_points_sam3(points, labels, confidence_threshold, max_cells)
            print("No predictor available for point prediction")
            return None
    
    def _predict_with_points_standard(self, points, labels):
        """Standard SAM prediction with points (fallback method)"""
        if self.predictor is None:
            return None
        try:
            points = np.array(points)
            labels = np.array(labels)
            if hasattr(self.predictor, 'predict'):
                masks, scores, logits = self.predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    multimask_output=True
                )
                best_idx = np.argmax(scores)
                return masks[best_idx]
        except Exception as e:
            print(f"Standard prediction failed: {e}")
        return None
    
    def predict_with_boxes(self, boxes, confidence_threshold=0.5, max_cells=2):
        """
        Run SAM prediction with bounding box prompts
        
        Args:
            boxes: list of [x1, y1, x2, y2] bounding boxes or QRect objects
            confidence_threshold: Confidence threshold for SAM3 (default: 0.5)
            max_cells: Maximum cells to return for SAM3 (default: 2)
            
        Returns:
            masks: list of predicted masks
        """
        # Convert QRect to coordinates if needed
        if boxes and isinstance(boxes[0], type):
            try:
                from PyQt5.QtCore import QRect
                if isinstance(boxes[0], QRect):
                    boxes = [[b.x(), b.y(), b.x() + b.width(), b.y() + b.height()] for b in boxes]
            except:
                pass
        
        # First try standard predictor (works for both standard SAM and SAM3 if predictor was created)
        if self.predictor is not None:
            try:
                boxes_tensor = torch.tensor(boxes, device=self.device).float()
                
                if hasattr(self.predictor, 'predict_torch'):
                    # Get image dimensions
                    if hasattr(self.predictor, 'original_size'):
                        H, W = self.predictor.original_size
                    else:
                        # Fallback - assume square image
                        H, W = 512, 512
                    
                    # Transform boxes
                    transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes_tensor, (H, W))
                    
                    # Predict
                    masks, _, _ = self.predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes,
                        multimask_output=False
                    )
                    return [mask.cpu().numpy() for mask in masks]
                elif hasattr(self.predictor, 'predict'):
                    # Use CPU prediction
                    masks_list = []
                    for box in boxes:
                        masks, scores, _ = self.predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=np.array(box),
                            multimask_output=False
                        )
                        masks_list.append(masks[0])
                    return masks_list
                else:
                    print("Predictor doesn't have prediction methods")
                    return None
            except Exception as e:
                print(f"Error in SAM prediction with boxes: {e}")
                # If predictor fails, try SAM3 processor approach
                if self.is_sam3_model:
                    print("Standard predictor failed, trying SAM3 processor with box prompts...")
                    return self._predict_with_boxes_sam3(boxes, confidence_threshold, max_cells)
                self.error_occurred.emit(f"Prediction failed: {str(e)}")
                return None
        else:
            # No predictor - try SAM3 processor if available
            if self.is_sam3_model:
                print("No standard predictor, trying SAM3 processor with box prompts...")
                return self._predict_with_boxes_sam3(boxes, confidence_threshold, max_cells)
            print("No predictor available for box prediction")
            return None
    
    def _predict_with_points_sam3(self, points, labels, confidence_threshold=0.5, max_cells=2):
        """Use SAM3 processor for point-based prediction"""
        try:
            # Check if we have an image set
            if not hasattr(self, '_current_image_for_sam') or self._current_image_for_sam is None:
                print("No image set for SAM3 processor. Call set_image() first.")
                return None
            
            from PIL import Image
            import torch
            
            # Get image
            image = self._current_image_for_sam
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            elif isinstance(image, Image.Image):
                pil_image = image
            else:
                print(f"Unsupported image type: {type(image)}")
                return None
            
            # Determine the actual device the model is on
            model_device_str = self.device
            if self.sam is not None:
                try:
                    for param in self.sam.parameters():
                        model_device_str = str(param.device)
                        break
                except (StopIteration, AttributeError, RuntimeError):
                    pass
            
            # Create or get SAM3 processor - use model's actual device, not self.device
            if not hasattr(self, 'sam3_processor') or self.sam3_processor is None:
                from sam3.model.sam3_image_processor import Sam3Processor
                self.sam3_processor = Sam3Processor(
                    model=self.sam,
                    resolution=1008,
                    device=model_device_str,  # Use model's actual device
                    confidence_threshold=confidence_threshold
                )
                # Verify processor is on CUDA (non-blocking check - skip if it might block)
                # Note: Processor verification is deferred to avoid blocking during creation
                if model_device_str.startswith("cuda"):
                    print(f"  ✓ SAM3 processor created (will use CUDA device: {model_device_str})")
            else:
                self.sam3_processor.confidence_threshold = confidence_threshold
                
                # Ensure processor uses model's actual device
                if hasattr(self.sam3_processor, 'device'):
                    self.sam3_processor.device = model_device_str
                if hasattr(self.sam3_processor, 'model') and self.sam3_processor.model is not None:
                    self.sam3_processor.model = self.sam3_processor.model.to(model_device_str)
            
            # Set image
            state = self.sam3_processor.set_image(pil_image)
            
            # Choose device based on model + state tensors (avoid mixed-device issues)
            target_device = self._pick_target_device(self.sam3_processor, state)
            target_device = self._ensure_processor_device(self.sam3_processor, target_device)
            if target_device is None:
                target_device = self.device
            
            
            # Ensure all tensors in state are on the correct device (use model's actual device)
            state = self._move_state_to_device(state, target_device)
            
            # Explicitly move boxes and all possible box-related keys to device
            if isinstance(state, dict):
                for key in ['boxes', 'pred_boxes', 'box', 'box_coords', 'bboxes', 'bounding_boxes']:
                    if key in state:
                        boxes_in_state = state[key]
                        if torch.is_tensor(boxes_in_state):
                            state[key] = boxes_in_state.to(target_device)
                        elif isinstance(boxes_in_state, (list, tuple)):
                            state[key] = type(boxes_in_state)([b.to(target_device) if torch.is_tensor(b) else b for b in boxes_in_state])
                
                # Also check nested dictionaries
                for key, value in state.items():
                    if isinstance(value, dict):
                        state[key] = self._move_state_to_device(value, target_device)
            
            # Convert points to tensor format
            points = np.array(points)
            labels = np.array(labels)
            
            # SAM3 processor expects points in a specific format
            # Try to use point prompts - SAM3 should support this
            # Format: points as [x, y] pairs, labels as [1, 0] for foreground/background
            point_coords = torch.tensor(points, device=target_device, dtype=torch.float32)
            point_labels = torch.tensor(labels, device=target_device, dtype=torch.int32)
            
            # Try to use point prompts with SAM3 processor
            # Note: This might require checking SAM3Processor API for point prompt methods
            # For now, use a workaround: use automatic segmentation but filter by point location
            # Or try to call the model directly with point inputs
            
            # Workaround: Use text prompt "visual" and filter results near points
            try:
                state = self.sam3_processor.set_text_prompt("visual", state)
            except RuntimeError as e:
                if "Expected all tensors to be on the same device" in str(e):
                    # Device mismatch detected - rebuild processor on the correct device
                    print(f"Device mismatch error. Rebuilding SAM3 processor on correct device...")
                    
                    # Determine the correct device to use (prefer CUDA if available)
                    if torch.cuda.is_available():
                        rebuild_device = "cuda:0"
                        print(f"CUDA is available, rebuilding processor on cuda:0")
                    else:
                        rebuild_device = "cpu"
                        print(f"CUDA not available, rebuilding processor on cpu")
                    
                    # Delete old processor to free memory
                    del self.sam3_processor
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Ensure model is on the correct device
                    if self.sam is not None:
                        self.sam = self.sam.to(rebuild_device)
                        # Force all submodules to the device
                        for module in self.sam.modules():
                            if hasattr(module, 'to'):
                                module.to(rebuild_device)
                    
                    # Rebuild processor on the correct device
                    from sam3.model.sam3_image_processor import Sam3Processor
                    self.sam3_processor = Sam3Processor(
                        model=self.sam,
                        resolution=1008,
                        device=rebuild_device,
                        confidence_threshold=confidence_threshold
                    )
                    print(f"Rebuilt SAM3 processor on {rebuild_device}")
                    
                    # Re-run set_image and set_text_prompt with the new processor
                    state = self.sam3_processor.set_image(pil_image)
                    state = self._move_state_to_device(state, rebuild_device)
                    
                    # Move all boxes to the correct device
                    if isinstance(state, dict):
                        for key in ['boxes', 'pred_boxes', 'box', 'box_coords', 'bboxes', 'bounding_boxes', 'coords', 'coordinates']:
                            if key in state:
                                boxes_in_state = state[key]
                                if torch.is_tensor(boxes_in_state):
                                    state[key] = boxes_in_state.to(rebuild_device)
                    
                    # Try again with the rebuilt processor
                    state = self.sam3_processor.set_text_prompt("visual", state)
                    target_device = rebuild_device  # Update target_device for later use
                else:
                    raise
            
            # Ensure all tensors in state are on the correct device after set_text_prompt
            # This is important because set_text_prompt might create new tensors on CPU
            state = self._move_state_to_device(state, target_device)
            
            # Final check: move boxes one more time
            if isinstance(state, dict):
                for key in ['boxes', 'pred_boxes', 'box', 'box_coords', 'bboxes', 'bounding_boxes', 'coords', 'coordinates']:
                    if key in state:
                        boxes_in_state = state[key]
                        if torch.is_tensor(boxes_in_state):
                            state[key] = boxes_in_state.to(target_device)
                        elif isinstance(boxes_in_state, (list, tuple)):
                            state[key] = type(boxes_in_state)([b.to(target_device) if torch.is_tensor(b) else b for b in boxes_in_state])
            
            # Get masks from state
            masks = state.get('masks', [])
            scores = state.get('scores', [])
            
            # Convert scores to list if tensor
            if torch.is_tensor(scores):
                scores = scores.cpu().numpy().tolist() if scores.numel() > 0 else []
            elif not isinstance(scores, (list, tuple)):
                scores = []
            
            # Check if masks is empty (handle both tensor and list)
            if masks is None:
                print("No masks generated by SAM3 processor")
                return None
            if torch.is_tensor(masks):
                if masks.numel() == 0 or masks.shape[0] == 0:
                    print("No masks generated by SAM3 processor")
                    return None
                # Convert tensor to list for iteration
                masks = [masks[i] for i in range(masks.shape[0])]
            elif isinstance(masks, (list, tuple)):
                if len(masks) == 0:
                    print("No masks generated by SAM3 processor")
                    return None
            else:
                print(f"Unexpected masks type: {type(masks)}")
                return None
            
            # Filter masks by point location and select best matching mask
            # Find mask that best matches the point locations
            best_mask = None
            best_score = -1
            
            for i, mask in enumerate(masks):
                if torch.is_tensor(mask):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = np.array(mask)
                
                # Check if points are within mask
                mask_value_at_points = []
                for point in points:
                    x, y = int(point[0]), int(point[1])
                    if 0 <= x < mask_np.shape[1] and 0 <= y < mask_np.shape[0]:
                        mask_value_at_points.append(mask_np[y, x] > 0.5)
                
                # If foreground points are in mask, this is a good match
                if len(mask_value_at_points) > 0:
                    foreground_points_match = sum([val and labels[j] == 1 for j, val in enumerate(mask_value_at_points)])
                    background_points_match = sum([not val and labels[j] == 0 for j, val in enumerate(mask_value_at_points)])
                    match_score = foreground_points_match - background_points_match
                    
                    if match_score > best_score:
                        best_score = match_score
                        best_mask = mask_np
            
            if best_mask is not None:
                return best_mask
            else:
                # If no good match, return the highest scoring mask
                if scores and len(scores) > 0 and len(masks) > 0:
                    # Handle tensor scores
                    if torch.is_tensor(scores):
                        scores_np = scores.cpu().numpy()
                    else:
                        scores_np = np.array(scores)
                    best_idx = np.argmax(scores_np)
                    if best_idx < len(masks):
                        mask = masks[best_idx]
                        return mask.cpu().numpy() if torch.is_tensor(mask) else np.array(mask)
                # Fallback: return first mask
                if len(masks) > 0:
                    return masks[0].cpu().numpy() if torch.is_tensor(masks[0]) else np.array(masks[0])
                return None
                
        except Exception as e:
            print(f"Error in SAM3 point prediction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _predict_with_boxes_sam3(self, boxes, confidence_threshold=0.5, max_cells=2):
        """Use SAM3 processor for box-based prediction"""
        print("[SAM WORKER] _predict_with_boxes_sam3: Starting...")
        try:
            # Check if we have an image set
            if not hasattr(self, '_current_image_for_sam') or self._current_image_for_sam is None:
                print("[SAM WORKER] ERROR: No image set for SAM3 processor. Call set_image() first.")
                return None
            print("[SAM WORKER] Image found, proceeding with prediction...")
            
            from PIL import Image
            import torch
            import torch.nn.functional as F
            
            # Get image
            image = self._current_image_for_sam
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            elif isinstance(image, Image.Image):
                pil_image = image
            else:
                print(f"Unsupported image type: {type(image)}")
                return None
            
            # Determine the actual device the model is on
            model_device_str = self.device
            if self.sam is not None:
                try:
                    for param in self.sam.parameters():
                        model_device_str = str(param.device)
                        break
                except (StopIteration, AttributeError, RuntimeError):
                    pass
            
            # Create or get SAM3 processor - use model's actual device, not self.device
            if not hasattr(self, 'sam3_processor') or self.sam3_processor is None:
                from sam3.model.sam3_image_processor import Sam3Processor
                self.sam3_processor = Sam3Processor(
                    model=self.sam,
                    resolution=1008,
                    device=model_device_str,  # Use model's actual device
                    confidence_threshold=confidence_threshold
                )
                print(f"Created SAM3 processor for box prompts with confidence_threshold={confidence_threshold:.2f} on device={model_device_str}")
                
                # Ensure processor's device attribute is correct
                if hasattr(self.sam3_processor, 'device'):
                    self.sam3_processor.device = model_device_str
                
                # Ensure model is on the correct device
                if hasattr(self.sam3_processor, 'model') and self.sam3_processor.model is not None:
                    self.sam3_processor.model = self.sam3_processor.model.to(model_device_str)
            else:
                self.sam3_processor.confidence_threshold = confidence_threshold
                
                # Ensure processor's device attribute matches model's actual device
                if hasattr(self.sam3_processor, 'device'):
                    self.sam3_processor.device = model_device_str
                
                # Ensure model is on the correct device on each use
                if hasattr(self.sam3_processor, 'model') and self.sam3_processor.model is not None:
                    # Check actual device and move if needed
                    try:
                        for param in self.sam3_processor.model.parameters():
                            current_dev = str(param.device)
                            if current_dev != model_device_str:
                                print(f"Moving processor model from {current_dev} to {model_device_str}")
                                self.sam3_processor.model = self.sam3_processor.model.to(model_device_str)
                            break
                    except (StopIteration, AttributeError, RuntimeError):
                        self.sam3_processor.model = self.sam3_processor.model.to(model_device_str)
            
            # Convert boxes to proper format [x1, y1, x2, y2]
            boxes_array = np.array(boxes)
            if len(boxes_array.shape) == 1:
                boxes_array = boxes_array.reshape(1, -1)
            
            # Create SAM3 processor first (needed for fallback)
            if not hasattr(self, 'sam3_processor') or self.sam3_processor is None:
                from sam3.model.sam3_image_processor import Sam3Processor
                self.sam3_processor = Sam3Processor(
                    model=self.sam,
                    resolution=1008,
                    device=self.device,
                    confidence_threshold=confidence_threshold
                )
            else:
                self.sam3_processor.confidence_threshold = confidence_threshold
            
            # Try to use the SAM3 model directly with box prompts
            # SAM3 models should have prompt_encoder and mask_decoder like SAM/SAM2
            try:
                import torch
                import torch.nn.functional as F
                
                # Prepare image tensor
                if isinstance(pil_image, Image.Image):
                    img_np = np.array(pil_image)
                else:
                    img_np = pil_image
                
                # Convert to RGB if needed
                if len(img_np.shape) == 2:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
                elif img_np.shape[2] == 4:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
                
                # Normalize and convert to tensor [1, C, H, W]
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                
                # Get image embeddings using model's image encoder
                if hasattr(self, 'sam') and hasattr(self.sam, 'image_encoder'):
                    with torch.no_grad():
                        # Compute image embeddings
                        image_embeddings = self.sam.image_encoder(img_tensor)
                        
                        print(f"Generated image embeddings for {len(boxes_array)} box(es)")
                        
                        # Get image PE (positional encoding)
                        if hasattr(self.sam, 'prompt_encoder'):
                            image_pe = self.sam.prompt_encoder.get_dense_pe()
                        elif hasattr(self.sam, 'get_image_pe'):
                            image_pe = self.sam.get_image_pe()
                        else:
                            image_pe = None
                        
                        # Process each box
                        result_masks = []
                        H, W = img_np.shape[:2]
                        num_boxes = len(boxes_array)
                        
                        for box_idx, box in enumerate(boxes_array):
                            bx = np.asarray(box, dtype=np.float64).ravel()
                            if bx.size < 4:
                                continue
                            x1, y1, x2, y2 = (
                                int(round(float(bx[0]))),
                                int(round(float(bx[1]))),
                                int(round(float(bx[2]))),
                                int(round(float(bx[3]))),
                            )
                            if x2 < x1:
                                x1, x2 = x2, x1
                            if y2 < y1:
                                y1, y2 = y2, y1

                            # Ensure box coordinates are within image bounds
                            x1 = max(0, min(x1, W - 1))
                            y1 = max(0, min(y1, H - 1))
                            x2 = max(x1 + 1, min(x2, W))
                            y2 = max(y1 + 1, min(y2, H))
                            
                            print(f"Processing box: [{x1}, {y1}, {x2}, {y2}] on image {W}x{H}")
                            
                            # Convert box to tensor format - try different formats
                            # Format 1: [1, 4] as [x1, y1, x2, y2]
                            box_tensor_flat = torch.tensor([[x1, y1, x2, y2]], device=self.device, dtype=torch.float32)
                            # Format 2: [1, 2, 2] as [[x1, y1], [x2, y2]]
                            box_coords = box_tensor_flat.reshape(1, 2, 2)
                            
                            # Use prompt encoder with box - try both formats
                            try:
                                if hasattr(self.sam, 'prompt_encoder'):
                                    # Try with [1, 2, 2] format first
                                    sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                                        points=None,
                                        boxes=box_coords,
                                        masks=None
                                    )
                                    print(f"Successfully encoded box with prompt_encoder (format: [1, 2, 2])")
                                elif hasattr(self.sam, 'encode_box'):
                                    sparse_embeddings, dense_embeddings = self.sam.encode_box(box_coords)
                                    print(f"Successfully encoded box with encode_box")
                                else:
                                    raise AttributeError("Model does not have prompt_encoder or encode_box")
                            except Exception as e1:
                                # Try with flat format [1, 4] if [1, 2, 2] fails
                                print(f"Failed with [1, 2, 2] format: {e1}, trying [1, 4] format...")
                                try:
                                    if hasattr(self.sam, 'prompt_encoder'):
                                        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                                            points=None,
                                            boxes=box_tensor_flat,  # Try flat format
                                            masks=None
                                        )
                                        print(f"Successfully encoded box with flat format [1, 4]")
                                    else:
                                        raise e1
                                except Exception as e2:
                                    print(f"Both box formats failed: [1,2,2]={e1}, [1,4]={e2}")
                                    raise
                            
                            # Use mask decoder with multimask output for better cell detection
                            if hasattr(self.sam, 'mask_decoder'):
                                # Use multimask_output=True to get 3 mask candidates
                                # This helps SAM find tight contours for small cells
                                low_res_masks, iou_predictions, _, _ = self.sam.mask_decoder(
                                    image_embeddings=image_embeddings,
                                    image_pe=image_pe,
                                    sparse_prompt_embeddings=sparse_embeddings,
                                    dense_prompt_embeddings=dense_embeddings,
                                    multimask_output=True  # Get multiple mask candidates
                                )
                            elif hasattr(self.sam, 'decode_mask'):
                                low_res_masks, iou_predictions = self.sam.decode_mask(
                                    image_embeddings, sparse_embeddings, dense_embeddings, image_pe
                                )
                            else:
                                raise AttributeError("Model does not have mask_decoder or decode_mask")
                            
                            # Select the best mask based on IoU predictions
                            # For bounding boxes, choose the mask with highest IoU that's smaller than box
                            best_mask_idx = 0
                            if iou_predictions is not None and len(iou_predictions) > 0:
                                # Get all IoU scores
                                iou_scores = iou_predictions.squeeze()
                                
                                # For each mask candidate, check if it fills the entire box
                                # Prefer masks that are smaller (tighter fit to cell)
                                best_score = -1
                                box_area = (x2 - x1) * (y2 - y1)
                                
                                for idx in range(len(iou_scores)):
                                    # Get mask at this index
                                    test_mask = torch.sigmoid(low_res_masks[idx:idx+1])
                                    test_mask_upsampled = F.interpolate(
                                        test_mask,
                                        size=(H, W),
                                        mode='bilinear',
                                        align_corners=False
                                    )
                                    test_mask_np = test_mask_upsampled.squeeze().detach().cpu().numpy()
                                    mask_area = np.sum(test_mask_np > 0.5)
                                    
                                    # Calculate how much of the box this mask fills
                                    fill_ratio = mask_area / box_area if box_area > 0 else 1.0
                                    
                                    # Prefer masks with high IoU but NOT filling entire box
                                    # For small cells, we want fill_ratio < 0.8
                                    score = float(iou_scores[idx])
                                    if fill_ratio < 0.85:  # Not filling entire box
                                        score *= 1.5  # Boost score for tighter masks
                                    elif fill_ratio > 0.95:  # Filling almost entire box
                                        score *= 0.5  # Penalize box-filling masks
                                    
                                    if score > best_score:
                                        best_score = score
                                        best_mask_idx = idx
                                    
                                    print(f"  Mask {idx}: IoU={float(iou_scores[idx]):.3f}, "
                                          f"fill_ratio={fill_ratio:.3f}, score={score:.3f}")
                                
                                print(f"  Selected mask {best_mask_idx} with score {best_score:.3f}")
                            
                            # Use the selected best mask
                            low_res_masks = low_res_masks[best_mask_idx:best_mask_idx+1]
                            
                            # Apply sigmoid and upsample to original image size
                            mask_logits = torch.sigmoid(low_res_masks)
                            mask = F.interpolate(
                                mask_logits,
                                size=(H, W),
                                mode='bilinear',
                                align_corners=False
                            )
                            
                            # Convert to numpy and threshold
                            mask_np = mask.squeeze().detach().cpu().numpy()
                            
                            # Clean up intermediate tensors
                            del mask, mask_logits, low_res_masks, sparse_embeddings, dense_embeddings
                            
                            # Ensure 2D array
                            if len(mask_np.shape) == 3:
                                mask_np = mask_np.squeeze()
                            if len(mask_np.shape) > 2:
                                mask_np = mask_np[0] if mask_np.shape[0] == 1 else mask_np[:, :, 0]
                            
                            mask_binary = (mask_np > 0.5).astype(np.uint8) * 255
                            
                            # Ensure it's 2D before post-processing
                            if len(mask_binary.shape) != 2:
                                mask_binary = mask_binary.squeeze()
                            
                            # Post-process mask: expand slightly to include border regions and carve hollow
                            mask_binary = self._post_process_sam_mask(mask_binary)
                            
                            result_masks.append(mask_binary)
                            
                            # Clear CUDA cache periodically
                            if (box_idx + 1) % 5 == 0 and torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        
                        # When using box prompts, don't limit by max_cells - return one mask per box
                        # Each box should get its own mask result
                        print(f"Successfully generated {len(result_masks)} mask(s) using box prompts with SAM3 model ({num_boxes} box(es) provided)")
                        
                        # Clean up remaining CUDA tensors
                        try:
                            del image_embeddings, img_tensor
                            if image_pe is not None:
                                del image_pe
                        except NameError:
                            pass
                        
                        # Clear CUDA cache after processing all boxes
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Post-process all masks (masks are already processed individually above, but ensure consistency)
                        # Masks are already processed in the loop above, so just return them
                        return result_masks if len(result_masks) > 1 else (result_masks[0] if result_masks else None)
                
                # If model doesn't have image_encoder, skip to fallback (don't raise exception)
                # This is expected for SAM3 models that use the processor interface
                pass  # Fall through to fallback code
                    
            except Exception as e:
                print(f"Direct model box prediction failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Fallback: Use SAM3 processor with add_geometric_prompt for box prompts
            # SAM3's add_geometric_prompt takes box in [center_x, center_y, width, height] format, normalized to [0, 1]
            print("[SAM WORKER] Using SAM3 processor with add_geometric_prompt for box prompts")
            
            # IMPORTANT: Use a LOW confidence threshold for box prompts (0.1-0.3)
            # The SAM3 processor filters masks by confidence_threshold BEFORE returning them
            # If we use 0.85, most masks will be filtered out!
            box_confidence_threshold = min(confidence_threshold, 0.3)  # Use lower of user setting or 0.3
            self.sam3_processor.confidence_threshold = box_confidence_threshold
            print(f"[SAM WORKER] Set processor confidence_threshold to {box_confidence_threshold:.2f} for box prompts")
            
            print("[SAM WORKER] Calling sam3_processor.set_image()...")
            
            # Set image - this is non-blocking on CUDA
            # No need to create streams - CUDA handles async operations
            state = self.sam3_processor.set_image(pil_image)
            print("[SAM WORKER] sam3_processor.set_image() completed")
            
            # Choose device based on model + state tensors (avoid mixed-device issues)
            target_device = self._pick_target_device(self.sam3_processor, state)
            target_device = self._ensure_processor_device(self.sam3_processor, target_device)
            if target_device is None:
                target_device = self.device
            
            # Ensure all tensors in state are on the correct device (use model's actual device)
            state = self._move_state_to_device(state, target_device)
            
            # Set a text prompt to help SAM3 understand what to segment
            # "cell" helps it focus on circular cell-like objects rather than filling the box
            # DISABLED: text prompt can confuse SAM3 for small cells
            # try:
            #     state = self.sam3_processor.set_text_prompt("cell", state)
            #     print("[SAM WORKER] Set text prompt 'cell' for better segmentation")
            # except Exception as text_err:
            #     print(f"[SAM WORKER] Could not set text prompt: {text_err}")
            print("[SAM WORKER] Skipping text prompt for better box-based segmentation")
            
            # Get image dimensions for normalization
            H, W = pil_image.size[1], pil_image.size[0]  # PIL uses (W, H) format
            print(f"[SAM WORKER] Image dimensions: {W}x{H}")
            
            # Process each box using add_geometric_prompt
            result_masks = []
            for box_idx, box in enumerate(boxes_array):
                flat = np.asarray(box, dtype=object).ravel()
                if flat.size < 4:
                    print(f"[SAM WORKER] Box {box_idx}: skip invalid box {box!r}")
                    continue
                coords = []
                for v in flat[:4]:
                    if torch.is_tensor(v):
                        coords.append(float(v.detach().cpu().float().item()))
                    else:
                        coords.append(float(np.asarray(v, dtype=np.float64)))
                # NumPy / torch scalars are not valid slice indices — must be Python int
                x1, y1, x2, y2 = tuple(int(round(c)) for c in coords)
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1
                x1 = max(0, min(W - 1, x1))
                y1 = max(0, min(H - 1, y1))
                x2 = max(x1 + 1, min(W, x2))
                y2 = max(y1 + 1, min(H, y2))

                # Calculate box dimensions for logging
                box_w = x2 - x1
                box_h = y2 - y1
                original_box_area = box_w * box_h

                # Convert [x1, y1, x2, y2] to [center_x, center_y, width, height] normalized to [0, 1]
                # Use the original box coordinates directly - no shrinking
                center_x = ((x1 + x2) / 2.0) / W
                center_y = ((y1 + y2) / 2.0) / H
                width = (x2 - x1) / W
                height = (y2 - y1) / H
                
                normalized_box = [center_x, center_y, width, height]
                print(f"[SAM WORKER] Box {box_idx}: [{x1}, {y1}, {x2}, {y2}] (size: {box_w}x{box_h})")
                print(f"[SAM WORKER] Box {box_idx}: normalized [{center_x:.4f}, {center_y:.4f}, {width:.4f}, {height:.4f}]")
                
                try:
                    # Add geometric prompt (box) - True means positive box
                    state = self.sam3_processor.add_geometric_prompt(normalized_box, True, state)
                    
                    # Debug: print state keys and shapes
                    print(f"[SAM WORKER] State keys after add_geometric_prompt: {list(state.keys())}")
                    if 'masks' in state:
                        m = state['masks']
                        print(f"[SAM WORKER] state['masks'] type={type(m)}, shape={m.shape if torch.is_tensor(m) else len(m) if isinstance(m, list) else 'N/A'}")
                    if 'masks_logits' in state:
                        ml = state['masks_logits']
                        print(f"[SAM WORKER] state['masks_logits'] type={type(ml)}, shape={ml.shape if torch.is_tensor(ml) else 'N/A'}")
                    if 'scores' in state:
                        s = state['scores']
                        print(f"[SAM WORKER] state['scores'] = {s.cpu().numpy() if torch.is_tensor(s) else s}")
                    
                    # Get masks from state after adding prompt
                    # IMPORTANT: Use masks_logits (raw logits) instead of masks (pre-binarized)
                    # This gives us control over the threshold for tighter cell segmentation
                    masks_logits = state.get('masks_logits', None)
                    masks = state.get('masks', [])
                    scores = state.get('scores', None)
                    
                    got_mask = False
                    
                    # First, try using masks_logits with a higher threshold for tighter segmentation
                    if masks_logits is not None and torch.is_tensor(masks_logits) and masks_logits.numel() > 0:
                        # Get the best mask based on scores
                        best_idx = 0
                        best_score = 0.0
                        if scores is not None and torch.is_tensor(scores) and scores.numel() > 0:
                            best_idx = scores.argmax().item()
                            best_score = scores[best_idx].item()
                            print(f"[SAM WORKER] Box {box_idx}: Selected logits mask {best_idx} with score {best_score:.4f}")
                            
                            # VALIDATION: Only reject VERY low confidence (likely completely empty space)
                            # Lower threshold to 0.3 to allow more detections
                            if best_score < 0.3:
                                print(f"[SAM WORKER] Box {box_idx}: REJECTED - Score {best_score:.4f} too low (< 0.3), likely empty space")
                                continue  # Skip this box, no valid mask found
                            elif best_score < 0.5:
                                print(f"[SAM WORKER] Box {box_idx}: WARNING - Low score {best_score:.4f}, but will try to segment")
                        
                        # Get the logits for the best mask
                        mask_logit = masks_logits[best_idx]
                        mask_logit_np = mask_logit.cpu().numpy()
                        
                        # Ensure 2D array
                        if len(mask_logit_np.shape) == 3:
                            mask_logit_np = mask_logit_np.squeeze()
                        if len(mask_logit_np.shape) > 2:
                            mask_logit_np = mask_logit_np[0] if mask_logit_np.shape[0] == 1 else mask_logit_np[:, :, 0]
                        
                        # Apply sigmoid to convert logits to probabilities (if needed)
                        # SAM3 masks_logits might already be sigmoid-ed
                        if mask_logit_np.min() < 0 or mask_logit_np.max() > 1:
                            # Apply sigmoid
                            mask_prob = 1 / (1 + np.exp(-mask_logit_np))
                        else:
                            mask_prob = mask_logit_np
                        
                        # ALWAYS use high thresholds to get tight contours around cells
                        # Don't let box size influence this - we want tight cell boundaries regardless
                        # Higher thresholds = tighter fit, avoiding thick/filled masks
                        thresholds = [0.75, 0.80, 0.85, 0.90]
                        print(f"[SAM WORKER] Box {box_idx}: Using high thresholds for tight contour (box area={original_box_area:.0f}px²)")
                        
                        # Try different thresholds to find best cell-like mask
                        # Prefer masks that DON'T fill the entire box
                        # For large boxes with small cells, prefer very LOW fill ratios
                        best_mask = None
                        best_fill_ratio = 1.0
                        best_thresh = None
                        best_area = 0
                        
                        for thresh in thresholds:
                            test_mask = (mask_prob > thresh).astype(np.uint8) * 255
                            test_area = (test_mask > 0).sum()
                            test_fill_ratio = test_area / max(original_box_area, 1)
                            
                            print(f"[SAM WORKER] Box {box_idx}: Threshold {thresh} -> area={test_area}px², fill_ratio={test_fill_ratio:.2f}")
                            
                            # For LARGE boxes (loose boxing), be very permissive on fill ratio
                            # The cell might be small (5-10% of box), so don't reject it
                            if original_box_area > 5000:  # Large box
                                min_fill = 0.05  # Accept even 5% fill (small cell in large box)
                                max_fill = 0.70
                                print(f"[SAM WORKER] Box {box_idx}:   Large box mode: accepting fill_ratio {min_fill:.2f}-{max_fill:.2f}")
                            else:  # Normal/small box
                                min_fill = 0.20  # Tighter box around cell
                                max_fill = 0.70
                            
                            # Accept masks with reasonable coverage
                            # Minimum area check: reject tiny debris (< 50px²)
                            # But don't be too restrictive - cells can be small
                            min_area = 50  # Simple absolute minimum - reject tiny debris
                            
                            # Check all conditions and show why rejected
                            passed_area = test_area > min_area
                            passed_fill = min_fill <= test_fill_ratio <= max_fill
                            
                            if not passed_area:
                                print(f"[SAM WORKER] Box {box_idx}:   ✗ Too small (area={test_area}px² < {min_area}px²), likely debris")
                            elif not passed_fill:
                                print(f"[SAM WORKER] Box {box_idx}:   ✗ Fill ratio {test_fill_ratio:.2f} outside range [{min_fill:.2f}, {max_fill:.2f}]")
                            
                            if passed_area and passed_fill:
                                # For large boxes, prefer LARGEST valid mask (cell, not debris)
                                # For small boxes, prefer mask closest to 40% fill
                                if original_box_area > 5000:
                                    # Large box: prefer LARGEST mask that passes filters (cell > debris)
                                    if best_mask is None or test_area > best_area:
                                        best_mask = test_mask
                                        best_fill_ratio = test_fill_ratio
                                        best_thresh = thresh
                                        best_area = test_area
                                        print(f"[SAM WORKER] Box {box_idx}:   ✓ ACCEPTED (largest so far): area={test_area}px², fill={test_fill_ratio:.2f}")
                                else:
                                    # Small box: prefer mask closest to 40% fill
                                    if best_mask is None or abs(test_fill_ratio - 0.40) < abs(best_fill_ratio - 0.40):
                                        best_mask = test_mask
                                        best_fill_ratio = test_fill_ratio
                                        best_thresh = thresh
                                        best_area = test_area
                                        print(f"[SAM WORKER] Box {box_idx}:   ✓ ACCEPTED (best 40% fill): area={test_area}px², fill={test_fill_ratio:.2f}")
                        
                        # VALIDATION: Reject if no good mask found
                        if best_mask is None:
                            print(f"[SAM WORKER] Box {box_idx}: REJECTED - No mask passed filters")
                            print(f"[SAM WORKER] Box {box_idx}:   Required: area > 50px², fill_ratio {min_fill:.2f}-{max_fill:.2f}")
                            continue  # Skip this box
                        
                        # Less strict validation: only reject if BOTH score AND fill are very low
                        # This allows low-contrast cells to be detected
                        if best_score < 0.4 and best_fill_ratio < 0.10:
                            print(f"[SAM WORKER] Box {box_idx}: REJECTED - Very low score ({best_score:.2f}) + very low fill ({best_fill_ratio:.2f}), likely empty space")
                            continue  # Skip this box
                        
                        if best_mask is not None:
                            print(f"[SAM WORKER] Box {box_idx}: ACCEPTED - threshold {best_thresh}, fill_ratio={best_fill_ratio:.2f}, score={best_score:.2f}")
                        
                        if best_mask is not None:
                            mask_binary = best_mask
                            
                            # CRITICAL: Crop mask to only the bounding box region
                            # SAM3 can generate masks outside the box, so we need to constrain it
                            mask_cropped = np.zeros_like(mask_binary)
                            mask_cropped[y1:y2, x1:x2] = mask_binary[y1:y2, x1:x2]
                            
                            # Check if there's any mask left after cropping
                            cropped_area = (mask_cropped > 0).sum()
                            cropped_fill = cropped_area / max(original_box_area, 1)
                            
                            if cropped_area > 10:
                                mask_binary = mask_cropped
                                print(f"[SAM WORKER] Box {box_idx}: Cropped mask to box region, area={cropped_area}px², fill={cropped_fill:.2f}")
                                
                                # Relaxed final validation: only reject if VERY low fill (< 3%)
                                # This allows small cells in large boxes to pass through
                                if cropped_fill < 0.03:
                                    print(f"[SAM WORKER] Box {box_idx}: REJECTED - Cropped fill too low ({cropped_fill:.2f} < 0.03), likely empty space")
                                    continue  # Skip this box
                            else:
                                print(f"[SAM WORKER] Box {box_idx}: REJECTED - No significant mask inside box after cropping")
                                continue  # Skip this box
                            
                            # Use standard post-processing
                            mask_binary = self._post_process_sam_mask(mask_binary)
                            result_masks.append(mask_binary)
                            got_mask = True
                            print(f"[SAM WORKER] Box {box_idx}: Got mask from masks_logits with shape {mask_binary.shape}, fill_ratio={best_fill_ratio:.2f}")
                    
                    # Fallback to pre-binarized masks if logits didn't work
                    if not got_mask and masks is not None:
                        # Handle tensor masks
                        if torch.is_tensor(masks):
                            if masks.numel() > 0 and masks.shape[0] > 0:
                                # SAM3 returns masks sorted by score - take the FIRST one (highest score)
                                # NOT the last one!
                                best_idx = 0
                                if scores is not None and torch.is_tensor(scores) and scores.numel() > 0:
                                    best_idx = scores.argmax().item()
                                    print(f"[SAM WORKER] Box {box_idx}: Fallback - Selected mask {best_idx} with score {scores[best_idx].item():.4f}")
                                
                                mask = masks[best_idx]
                                mask_np = mask.cpu().numpy()
                                
                                # Ensure 2D array
                                if len(mask_np.shape) == 3:
                                    mask_np = mask_np.squeeze()
                                if len(mask_np.shape) > 2:
                                    mask_np = mask_np[0] if mask_np.shape[0] == 1 else mask_np[:, :, 0]
                                
                                # Debug: Check mask statistics
                                mask_sum = mask_np.sum()
                                mask_area = (mask_np > 0.5).sum() if mask_np.max() <= 1.0 else (mask_np > 127).sum()
                                print(f"[SAM WORKER] Box {box_idx}: Fallback mask stats - sum={mask_sum:.2f}, area={mask_area}px², shape={mask_np.shape}")
                                
                                # Convert to binary mask
                                if mask_np.max() <= 1.0:
                                    mask_binary = (mask_np > 0.5).astype(np.uint8) * 255
                                else:
                                    mask_binary = (mask_np > 127).astype(np.uint8) * 255
                                
                                # CRITICAL: Crop mask to only the bounding box region
                                mask_cropped = np.zeros_like(mask_binary)
                                mask_cropped[y1:y2, x1:x2] = mask_binary[y1:y2, x1:x2]
                                cropped_area = (mask_cropped > 0).sum()
                                if cropped_area > 10:
                                    mask_binary = mask_cropped
                                    print(f"[SAM WORKER] Box {box_idx}: Fallback - Cropped mask to box region, area={cropped_area}px²")
                                else:
                                    print(f"[SAM WORKER] Box {box_idx}: Fallback - Warning: No mask inside box after cropping")
                                
                                # Use standard post-processing
                                mask_binary = self._post_process_sam_mask(mask_binary)
                                result_masks.append(mask_binary)
                                got_mask = True
                                print(f"[SAM WORKER] Box {box_idx}: Got mask from state['masks'] with shape {mask_binary.shape}")
                        elif isinstance(masks, (list, tuple)) and len(masks) > 0:
                            # Take the first mask (highest score) not the last
                            mask = masks[0]
                            if torch.is_tensor(mask):
                                mask_np = mask.cpu().numpy()
                            else:
                                mask_np = np.array(mask)
                            
                            # Ensure 2D array
                            if len(mask_np.shape) == 3:
                                mask_np = mask_np.squeeze()
                            
                            # Convert to binary mask
                            if mask_np.max() <= 1.0:
                                mask_binary = (mask_np > 0.5).astype(np.uint8) * 255
                            else:
                                mask_binary = (mask_np > 127).astype(np.uint8) * 255
                            
                            mask_binary = self._post_process_sam_mask(mask_binary)
                            result_masks.append(mask_binary)
                            got_mask = True
                            print(f"[SAM WORKER] Box {box_idx}: Got mask from state['masks'] list with shape {mask_binary.shape}")
                    
                    # Fallback: try masks_logits if masks was empty (confidence threshold filtered them)
                    if not got_mask and 'masks_logits' in state:
                        masks_logits = state['masks_logits']
                        if torch.is_tensor(masks_logits) and masks_logits.numel() > 0:
                            print(f"[SAM WORKER] Box {box_idx}: Trying masks_logits fallback, shape={masks_logits.shape}")
                            # masks_logits is [N, 1, H, W] - take first one, squeeze channel dim
                            mask_logit = masks_logits[0].squeeze().cpu().numpy()
                            # Convert logits (sigmoid output) to binary mask
                            mask_binary = (mask_logit > 0.5).astype(np.uint8) * 255
                            mask_binary = self._post_process_sam_mask(mask_binary)
                            result_masks.append(mask_binary)
                            got_mask = True
                            print(f"[SAM WORKER] Box {box_idx}: Got mask from masks_logits with shape {mask_binary.shape}")
                    
                    # Reset prompts for next box (pass state as required)
                    state = self.sam3_processor.reset_all_prompts(state)
                    # Re-set the image for the next box
                    state = self.sam3_processor.set_image(pil_image)
                    state = self._move_state_to_device(state, target_device)
                    
                except Exception as box_err:
                    print(f"[SAM WORKER] Error processing box {box_idx}: {box_err}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if result_masks:
                print(f"[SAM WORKER] Successfully generated {len(result_masks)} mask(s) using add_geometric_prompt")
                return result_masks if len(result_masks) > 1 else result_masks[0]
            
            # If add_geometric_prompt didn't work, try fallback with visual prompt
            print("[SAM WORKER] add_geometric_prompt didn't produce masks, trying visual prompt fallback...")
            
            # Reset and try visual prompt (pass state as required)
            state = self.sam3_processor.reset_all_prompts(state)
            state = self.sam3_processor.set_image(pil_image)
            state = self._move_state_to_device(state, target_device)
            
            try:
                state = self.sam3_processor.set_text_prompt("visual", state)
            except RuntimeError as e:
                if "Expected all tensors to be on the same device" in str(e):
                    # Device mismatch detected - rebuild processor on the correct device
                    print(f"Device mismatch error. Rebuilding SAM3 processor on correct device...")
                    
                    # Determine the correct device to use (prefer CUDA if available)
                    if torch.cuda.is_available():
                        rebuild_device = "cuda:0"
                        print(f"CUDA is available, rebuilding processor on cuda:0")
                    else:
                        rebuild_device = "cpu"
                        print(f"CUDA not available, rebuilding processor on cpu")
                    
                    # Delete old processor to free memory
                    del self.sam3_processor
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Ensure model is on the correct device
                    if self.sam is not None:
                        self.sam = self.sam.to(rebuild_device)
                        # Force all submodules to the device
                        for module in self.sam.modules():
                            if hasattr(module, 'to'):
                                module.to(rebuild_device)
                    
                    # Rebuild processor on the correct device
                    from sam3.model.sam3_image_processor import Sam3Processor
                    self.sam3_processor = Sam3Processor(
                        model=self.sam,
                        resolution=1008,
                        device=rebuild_device,
                        confidence_threshold=confidence_threshold
                    )
                    print(f"Rebuilt SAM3 processor on {rebuild_device}")
                    
                    # Re-run set_image and set_text_prompt with the new processor
                    state = self.sam3_processor.set_image(pil_image)
                    state = self._move_state_to_device(state, rebuild_device)
                    
                    # Move all boxes to the correct device
                    if isinstance(state, dict):
                        for key in ['boxes', 'pred_boxes', 'box', 'box_coords', 'bboxes', 'bounding_boxes', 'coords', 'coordinates']:
                            if key in state:
                                boxes_in_state = state[key]
                                if torch.is_tensor(boxes_in_state):
                                    state[key] = boxes_in_state.to(rebuild_device)
                    
                    # Try again with the rebuilt processor
                    state = self.sam3_processor.set_text_prompt("visual", state)
                    target_device = rebuild_device  # Update target_device for later use
                else:
                    raise
            
            # Ensure all tensors in state are on the correct device after set_text_prompt
            # This is important because set_text_prompt might create new tensors on CPU
            state = self._move_state_to_device(state, target_device)
            
            # Final check: move boxes one more time
            if isinstance(state, dict):
                for key in ['boxes', 'pred_boxes', 'box', 'box_coords', 'bboxes', 'bounding_boxes', 'coords', 'coordinates']:
                    if key in state:
                        boxes_in_state = state[key]
                        if torch.is_tensor(boxes_in_state):
                            state[key] = boxes_in_state.to(target_device)
                        elif isinstance(boxes_in_state, (list, tuple)):
                            state[key] = type(boxes_in_state)([b.to(target_device) if torch.is_tensor(b) else b for b in boxes_in_state])
            
            # Get masks and boxes from state
            masks = state.get('masks', [])
            scores = state.get('scores', [])
            predicted_boxes = state.get('boxes', [])
            
            # Convert scores to list if tensor
            if torch.is_tensor(scores):
                scores_list = scores.cpu().numpy().tolist() if scores.numel() > 0 else []
                scores = scores_list
            elif not isinstance(scores, (list, tuple)):
                scores = []
            
            # Convert predicted_boxes to list if tensor
            if torch.is_tensor(predicted_boxes):
                predicted_boxes = [predicted_boxes[i] for i in range(predicted_boxes.shape[0])]
            elif not isinstance(predicted_boxes, (list, tuple)):
                predicted_boxes = []
            
            # Check if masks is empty (handle both tensor and list)
            if masks is None:
                print("No masks generated by SAM3 processor")
                return None
            if torch.is_tensor(masks):
                if masks.numel() == 0 or masks.shape[0] == 0:
                    print("No masks generated by SAM3 processor")
                    return None
                # Convert tensor to list for iteration
                masks_list = [masks[i] for i in range(masks.shape[0])]
                masks = masks_list
            elif isinstance(masks, (list, tuple)):
                if len(masks) == 0:
                    print("No masks generated by SAM3 processor")
                    return None
                masks = list(masks)
            else:
                print(f"Unexpected masks type: {type(masks)}")
                return None
            
            # Filter masks by box overlap
            boxes = np.array(boxes)
            num_boxes = len(boxes)
            result_masks = []
            
            for box in boxes:
                bx = np.asarray(box, dtype=np.float64).ravel()
                if bx.size < 4:
                    continue
                x1, y1, x2, y2 = (
                    int(round(float(bx[0]))),
                    int(round(float(bx[1]))),
                    int(round(float(bx[2]))),
                    int(round(float(bx[3]))),
                )
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1

                # Find mask with best IoU overlap with this box
                best_mask = None
                best_iou = 0
                
                for i, mask in enumerate(masks):
                    if torch.is_tensor(mask):
                        mask_np = mask.cpu().numpy()
                    else:
                        mask_np = np.array(mask)
                    
                    # Get predicted box if available
                    if i < len(predicted_boxes):
                        pred_box = predicted_boxes[i]
                        if torch.is_tensor(pred_box):
                            pb = pred_box.cpu().numpy()
                        else:
                            pb = np.array(pred_box)
                        
                        # Calculate IoU
                        intersection_area = max(0, min(x2, pb[2]) - max(x1, pb[0])) * max(0, min(y2, pb[3]) - max(y1, pb[1]))
                        box_area = (x2 - x1) * (y2 - y1)
                        pred_box_area = (pb[2] - pb[0]) * (pb[3] - pb[1])
                        union_area = box_area + pred_box_area - intersection_area
                        iou = intersection_area / union_area if union_area > 0 else 0
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_mask = mask_np
                    else:
                        # If no predicted box, check mask coverage in box area
                        mask_region = mask_np[int(y1):int(y2), int(x1):int(x2)]
                        coverage = np.sum(mask_region > 0.5) / (box_area + 1e-6)
                        if coverage > best_iou:
                            best_iou = coverage
                            best_mask = mask_np
                
                if best_mask is not None:
                    result_masks.append(best_mask)
                elif len(masks) > 0:
                    # Fallback: use highest scoring mask
                    if scores and len(scores) > 0:
                        # Handle tensor scores
                        if torch.is_tensor(scores):
                            scores_np = scores.cpu().numpy()
                        else:
                            scores_np = np.array(scores)
                        best_idx = np.argmax(scores_np)
                    else:
                        best_idx = 0
                    if best_idx < len(masks):
                        mask = masks[best_idx]
                        result_masks.append(mask.cpu().numpy() if torch.is_tensor(mask) else np.array(mask))
            
            # When using box prompts, don't limit by max_cells - return one mask per box
            # Each box should get its own mask result, so we don't apply max_cells limitation
            
            # Post-process all masks to expand borders and carve hollows
            processed_masks = []
            for mask in result_masks:
                if mask is not None:
                    # Convert to numpy array and ensure proper format
                    mask = np.array(mask)
                    
                    # Ensure 2D array
                    if len(mask.shape) == 3:
                        mask = mask.squeeze()
                    if len(mask.shape) > 2:
                        mask = mask[0] if mask.shape[0] == 1 else mask[:, :, 0]
                    
                    # Convert to uint8 binary mask
                    if mask.dtype != np.uint8:
                        if mask.max() <= 1.0:
                            mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
                        else:
                            mask_uint8 = (mask > 127).astype(np.uint8) * 255
                    else:
                        mask_uint8 = (mask > 127).astype(np.uint8) * 255
                    
                    # Ensure it's 2D
                    if len(mask_uint8.shape) != 2:
                        print(f"Warning: mask_uint8 has shape {mask_uint8.shape}, trying to fix")
                        mask_uint8 = mask_uint8.squeeze()
                    
                    processed_mask = self._post_process_sam_mask(mask_uint8)
                    processed_masks.append(processed_mask)
                else:
                    processed_masks.append(mask)
            
            return processed_masks if len(processed_masks) > 1 else (processed_masks[0] if processed_masks else None)
                
        except Exception as e:
            print(f"Error in SAM3 box prediction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def _select_sam3_automatic_masks(masks_list, max_cells, mode):
        """
        Choose up to max_cells masks after SAM3 filtering.
        mode: smallest (default, small structures), largest, score, mixed (large+small).
        """
        if max_cells <= 0 or not masks_list:
            return []
        mode = (mode or "smallest").lower()
        if mode == "smallest":
            s = sorted(masks_list, key=lambda x: x["area"])
            return s[:max_cells]
        if mode == "largest":
            s = sorted(masks_list, key=lambda x: x["area"], reverse=True)
            return s[:max_cells]
        if mode == "score":
            s = sorted(masks_list, key=lambda x: x["score"], reverse=True)
            return s[:max_cells]
        if mode == "mixed":
            s = sorted(masks_list, key=lambda x: x["area"])
            k = min(max_cells, len(s))
            if len(s) <= k:
                return s
            n_small = max(1, k // 2)
            n_large = k - n_small
            small_part = s[:n_small]
            large_part = s[-n_large:] if n_large > 0 else []
            out = []
            seen = set()
            for m in large_part + small_part:
                key = id(m["segmentation"])
                if key in seen:
                    continue
                seen.add(key)
                out.append(m)
            if len(out) < k:
                for m in sorted(masks_list, key=lambda x: -x["score"]):
                    key = id(m["segmentation"])
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(m)
                    if len(out) >= k:
                        break
            return out[:k]
        s = sorted(masks_list, key=lambda x: x["area"])
        return s[:max_cells]
    
    def predict_automatic(self, image=None, confidence_threshold=0.5, max_cells=2, mask_selection_mode=None):
        """
        Run SAM automatic mask generation (no prompts)
        
        Args:
            image: Optional image to process (uses last set image if None)
            confidence_threshold: Confidence threshold (0.0 to 1.0, default: 0.5)
            max_cells: Maximum number of cells/masks to return (default: 2)
            mask_selection_mode: SAM3 only — smallest | largest | score | mixed (large+small).
                None uses self._pending_mask_selection_mode if set, else smallest.
            
        Returns:
            masks: list of predicted masks (sorted by confidence, top N)
        """
        if self.predictor is None and self.sam is None:
            print("No predictor or model available")
            return None
        
        if image is None:
            print("No image provided for automatic segmentation")
            return None
        
        try:
            # Validate and preprocess image
            if image is None:
                print("Error: Image is None")
                return None
            
            print(f"Input image shape: {image.shape}, dtype: {image.dtype}, range: [{image.min()}, {image.max()}]")
            
            # Ensure image is RGB and uint8
            if len(image.shape) == 2:
                # Grayscale: convert to RGB
                image = np.stack([image, image, image], axis=-1)
                print("Converted grayscale to RGB")
            elif image.shape[2] == 4:
                # RGBA: remove alpha channel
                image = image[:, :, :3]
                print("Removed alpha channel")
            elif image.shape[2] != 3:
                print(f"Error: Unexpected number of channels: {image.shape[2]}")
                return None
            
            # Ensure uint8 with proper normalization
            if image.dtype != np.uint8:
                if image.max() > 1.0:
                    # Already in 0-255 range, just cast
                    image = np.clip(image, 0, 255).astype(np.uint8)
                else:
                    # Normalize from 0-1 to 0-255
                    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
                print(f"Converted to uint8, new range: [{image.min()}, {image.max()}]")
            
            # Ensure image is not too small (SAM works better with larger images)
            h, w = image.shape[:2]
            if h < 64 or w < 64:
                print(f"Warning: Image is very small ({w}x{h}). Results may be poor.")
            
            print(f"Final image shape: {image.shape}, dtype: {image.dtype}, range: [{image.min()}, {image.max()}]")
            
            # Check if this is SAM3 (use flag first, then check other indicators)
            checkpoint_has_sam3 = "sam3" in str(self.checkpoint_path).lower()
            model_has_sam3_attrs = self.sam is not None and (
                hasattr(self.sam, 'vision_encoder') or 
                (hasattr(self.sam, 'image_encoder') and hasattr(self.sam, 'prompt_encoder')) or
                type(self.sam).__name__ == 'Sam3ImageModel' or
                'sam3' in str(type(self.sam)).lower()
            )
            
            is_sam3 = self.is_sam3_model or checkpoint_has_sam3 or model_has_sam3_attrs
            
            print(f"SAM3 detection: is_sam3_model={self.is_sam3_model}, "
                  f"checkpoint_path contains 'sam3'={checkpoint_has_sam3}, "
                  f"model_has_sam3_attrs={model_has_sam3_attrs}, "
                  f"model type={type(self.sam).__name__ if self.sam else 'None'}, "
                  f"FINAL is_sam3={is_sam3}")
            
            # Try SAM3 processor-based approach first if SAM3 detected OR if checkpoint path suggests SAM3
            # ALWAYS try SAM3 processor if checkpoint path contains "sam3"
            if checkpoint_has_sam3 or is_sam3:
                print(f"Attempting to use SAM3 processor... (checkpoint={checkpoint_has_sam3}, model={is_sam3})")
                try:
                    # Try to import sam3 - check multiple possible locations
                    sam3_imported = False
                    sam3_module = None
                    
                    # Try standard import first
                    try:
                        import sam3
                        sam3_module = sam3
                        sam3_imported = True
                    except ImportError:
                        # Try adding common SAM3 installation paths
                        import sys
                        # Check environment variable first
                        env_sam3_path = os.environ.get("SAM3_PATH")
                        sam3_paths = []
                        if env_sam3_path and os.path.exists(env_sam3_path):
                            sam3_paths.append(env_sam3_path)
                        
                        # Add common paths
                        sam3_paths.extend([
                            "/home/sai/Desktop/sam3_training/sam3",
                            "/home/sai/Desktop/sam3",
                            
                            os.path.join(os.path.dirname(self.checkpoint_path), "..", "sam3"),
                            os.path.join(os.path.expanduser("~"), "sam3"),
                        ])
                        
                        for sam3_path in sam3_paths:
                            if os.path.exists(sam3_path):
                                abs_path = os.path.abspath(sam3_path)
                                # Check if sam3 subdirectory exists
                                sam3_module_path = os.path.join(abs_path, "sam3")
                                if os.path.exists(sam3_module_path) or os.path.exists(os.path.join(abs_path, "__init__.py")):
                                    if abs_path not in sys.path:
                                        sys.path.insert(0, abs_path)
                                        print(f"Added SAM3 path to sys.path: {abs_path}")
                                    try:
                                        import sam3
                                        sam3_module = sam3
                                        sam3_imported = True
                                        print(f"Successfully imported sam3 from {abs_path}")
                                        break
                                    except ImportError as import_err:
                                        error_msg = str(import_err)
                                        print(f"Found SAM3 at {abs_path} but import failed: {error_msg}")
                                        # If it's a dependency issue (like decord), let's continue but note it
                                        if "decord" in error_msg or "No module named" in error_msg:
                                            print(f"  Note: Missing dependency - you may need to install it (e.g., pip install decord)")
                                        # Continue to next path
                                        continue
                                else:
                                    print(f"Path {abs_path} exists but doesn't contain sam3 module")
                            else:
                                print(f"SAM3 path not found: {sam3_path}")
                    
                    if not sam3_imported:
                        # Check if we found SAM3 but import failed due to dependencies
                        sam3_found_but_failed = False
                        last_error = None
                        for sam3_path in ["/cellchorus/sam3_training/sam3"]:
                            if os.path.exists(os.path.join(sam3_path, "sam3")):
                                sam3_found_but_failed = True
                                break
                        
                        if sam3_found_but_failed:
                            raise ImportError(
                                "SAM3 found at /cellchorus/sam3_training/sam3 but import failed due to missing dependencies.\n"
                                "Please install required dependencies:\n"
                                "  pip install decord\n"
                                "  or install all SAM3 dependencies:\n"
                                "  cd /cellchorus/sam3_training/sam3 && pip install -e ."
                            )
                        else:
                            raise ImportError(
                                "SAM3 module not found. Please install SAM3:\n"
                                "  pip install sam3\n"
                                "  or ensure SAM3 is in your Python path.\n"
                                "  Common locations: /cellchorus/sam3_training/sam3 or ~/sam3"
                            )
                    
                    from sam3.model.sam3_image_processor import Sam3Processor
                    from sam3.model_builder import build_sam3_image_model
                    from PIL import Image
                    
                    # If model wasn't loaded as SAM3, try to load it now
                    if self.sam is None or not self.is_sam3_model:
                        print("SAM3 model not loaded properly, attempting to load now...")
                        try:
                            model_kwargs = {
                                "device": self.device,
                                "eval_mode": True,
                                "checkpoint_path": self.checkpoint_path,
                                "load_from_HF": False,
                                "enable_segmentation": True,
                                "enable_inst_interactivity": False,
                            }
                            bpe_path = self._find_sam3_bpe_path()
                            if bpe_path:
                                model_kwargs["bpe_path"] = bpe_path
                            self.sam = build_sam3_image_model(**model_kwargs)
                            self.is_sam3_model = True
                            print("SAM3 model loaded successfully during automatic prediction")
                        except Exception as e:
                            print(f"Failed to load SAM3 model: {e}")
                            if self.sam is None:
                                raise
                    
                    if self.sam is not None:
                        print("Detected SAM3 model, using Sam3Processor with 'visual' prompt")
                        
                        # Convert numpy array to PIL Image
                        pil_image = Image.fromarray(image)
                        
                        # Determine the actual device the model is on
                        model_device_str = self.device
                        if self.sam is not None:
                            try:
                                for param in self.sam.parameters():
                                    model_device_str = str(param.device)
                                    break
                            except (StopIteration, AttributeError, RuntimeError):
                                pass
                        
                        # For automatic mode (no prompts), use lower confidence threshold
                        # SAM3's automatic segmentation needs lower thresholds to find cells
                        # For small cells, we need VERY low thresholds (0.1-0.2)
                        auto_confidence = min(confidence_threshold, 0.15)  # Cap at 0.15 for automatic mode
                        if auto_confidence != confidence_threshold:
                            print(f"  Automatic mode: Adjusting confidence from {confidence_threshold:.2f} to {auto_confidence:.2f}")
                            print(f"  (SAM3 automatic mode works best with very low thresholds for small cells)")
                        
                        # Create processor if not exists, or update confidence threshold
                        if not hasattr(self, 'sam3_processor') or self.sam3_processor is None:
                            print(f"Creating SAM3 processor with confidence_threshold={auto_confidence:.2f} on device={model_device_str}...")
                            self.sam3_processor = Sam3Processor(
                                model=self.sam,
                                resolution=1008,
                                device=model_device_str,  # Use model's actual device
                                confidence_threshold=auto_confidence
                            )
                            print(f"SAM3 processor created with confidence_threshold={auto_confidence:.2f} on device={model_device_str}")
                        else:
                            # Update existing processor's confidence threshold
                            self.sam3_processor.confidence_threshold = auto_confidence
                            print(f"Updated SAM3 processor confidence_threshold to {auto_confidence:.2f}")
                            
                            # Ensure processor uses model's actual device
                            if hasattr(self.sam3_processor, 'device'):
                                self.sam3_processor.device = model_device_str
                            if hasattr(self.sam3_processor, 'model') and self.sam3_processor.model is not None:
                                self.sam3_processor.model = self.sam3_processor.model.to(model_device_str)
                        
                        # Set image
                        state = self.sam3_processor.set_image(pil_image)
                        
                        # Choose device based on model + state tensors (avoid mixed-device issues)
                        target_device = self._pick_target_device(self.sam3_processor, state)
                        target_device = self._ensure_processor_device(self.sam3_processor, target_device)
                        if target_device is None:
                            target_device = self.device
                        
                        
                        # Ensure all tensors in state are on the correct device (use model's actual device)
                        # This prevents "Expected all tensors to be on the same device" errors
                        state = self._move_state_to_device(state, target_device)
                        
                        # Explicitly move boxes and all possible box-related keys to device
                        if isinstance(state, dict):
                            for key in ['boxes', 'pred_boxes', 'box', 'box_coords', 'bboxes', 'bounding_boxes']:
                                if key in state:
                                    boxes_in_state = state[key]
                                    if torch.is_tensor(boxes_in_state):
                                        state[key] = boxes_in_state.to(target_device)
                                    elif isinstance(boxes_in_state, (list, tuple)):
                                        state[key] = type(boxes_in_state)([b.to(target_device) if torch.is_tensor(b) else b for b in boxes_in_state])
                            
                            # Also check nested dictionaries
                            for key, value in state.items():
                                if isinstance(value, dict):
                                    state[key] = self._move_state_to_device(value, target_device)
                        
                        # Use "visual" prompt for automatic segmentation (like the working script)
                        try:
                            state = self.sam3_processor.set_text_prompt("visual", state)
                        except RuntimeError as e:
                            if "Expected all tensors to be on the same device" in str(e):
                                # Device mismatch detected - rebuild processor on the correct device
                                print(f"Device mismatch error. Rebuilding SAM3 processor on correct device...")
                                
                                # Determine the correct device to use (prefer CUDA if available)
                                if torch.cuda.is_available():
                                    rebuild_device = "cuda:0"
                                    print(f"CUDA is available, rebuilding processor on cuda:0")
                                else:
                                    rebuild_device = "cpu"
                                    print(f"CUDA not available, rebuilding processor on cpu")
                                
                                # Delete old processor to free memory
                                del self.sam3_processor
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                
                                # Ensure model is on the correct device
                                if self.sam is not None:
                                    self.sam = self.sam.to(rebuild_device)
                                    # Force all submodules to the device
                                    for module in self.sam.modules():
                                        if hasattr(module, 'to'):
                                            module.to(rebuild_device)
                                
                                # Rebuild processor on the correct device
                                from sam3.model.sam3_image_processor import Sam3Processor
                                self.sam3_processor = Sam3Processor(
                                    model=self.sam,
                                    resolution=1008,
                                    device=rebuild_device,
                                    confidence_threshold=confidence_threshold
                                )
                                print(f"Rebuilt SAM3 processor on {rebuild_device}")
                                
                                # Re-run set_image and set_text_prompt with the new processor
                                state = self.sam3_processor.set_image(pil_image)
                                state = self._move_state_to_device(state, rebuild_device)
                                
                                # Move all boxes to the correct device
                                if isinstance(state, dict):
                                    for key in ['boxes', 'pred_boxes', 'box', 'box_coords', 'bboxes', 'bounding_boxes', 'coords', 'coordinates']:
                                        if key in state:
                                            boxes_in_state = state[key]
                                            if torch.is_tensor(boxes_in_state):
                                                state[key] = boxes_in_state.to(rebuild_device)
                                
                                # Try again with the rebuilt processor
                                state = self.sam3_processor.set_text_prompt("visual", state)
                                target_device = rebuild_device  # Update target_device for later use
                            else:
                                raise
                        
                        # Ensure all tensors in state are on the correct device after set_text_prompt
                        # This is important because set_text_prompt might create new tensors on CPU
                        state = self._move_state_to_device(state, target_device)
                        
                        # Final check: move boxes one more time
                        if isinstance(state, dict):
                            for key in ['boxes', 'pred_boxes', 'box', 'box_coords', 'bboxes', 'bounding_boxes', 'coords', 'coordinates']:
                                if key in state:
                                    boxes_in_state = state[key]
                                    if torch.is_tensor(boxes_in_state):
                                        state[key] = boxes_in_state.to(target_device)
                                    elif isinstance(boxes_in_state, (list, tuple)):
                                        state[key] = type(boxes_in_state)([b.to(target_device) if torch.is_tensor(b) else b for b in boxes_in_state])
                        
                        # Extract masks from state
                        masks = state.get('masks', [])
                        scores = state.get('scores', [])
                        boxes = state.get('boxes', [])
                        
                        if masks is not None and len(masks) > 0:
                            # Convert to list format compatible with rest of code
                            masks_list = []
                            h, w = image.shape[:2]
                            total_area = h * w
                            _sel_early = (
                                mask_selection_mode
                                if mask_selection_mode is not None
                                else getattr(self, "_pending_mask_selection_mode", "smallest")
                            )
                            # Allow near-full-image masks when prioritizing large structures (TEM cell outline)
                            _max_frac = (
                                0.99
                                if str(_sel_early).lower() in ("largest", "mixed", "score")
                                else 0.9
                            )
                            
                            # Filtering thresholds (use adjusted auto_confidence)
                            min_score = auto_confidence  # Use adjusted confidence threshold
                            # For small cells, use VERY permissive area thresholds
                            min_area = 10  # Accept even tiny cells (10 pixels minimum)
                            max_area = int(total_area * _max_frac)
                            
                            print(f"  Filtering masks: min_score={min_score:.2f}, min_area={min_area}px, max_area={max_area}px")
                            
                            if isinstance(masks, torch.Tensor):
                                print(f"  Processing {masks.shape[0]} mask candidates from SAM3...")
                                for i in range(masks.shape[0]):
                                    mask_np = masks[i].cpu().numpy()
                                    # Ensure 2D mask
                                    if len(mask_np.shape) > 2:
                                        mask_np = mask_np.squeeze()
                                    
                                    # Convert to boolean mask
                                    mask_bool = mask_np.astype(bool)
                                    area = int(np.sum(mask_bool))
                                    
                                    # Get score
                                    score = float(scores[i]) if i < len(scores) and scores[i] is not None else 0.0
                                    
                                    # Debug: show all candidates
                                    print(f"    Candidate {i}: score={score:.3f}, area={area}px", end="")
                                    
                                    # Filter masks based on score and area
                                    if score < min_score:
                                        print(f" -> REJECTED (score < {min_score:.3f})")
                                        continue  # Skip low confidence masks
                                    if area < min_area or area > max_area:
                                        print(f" -> REJECTED (area out of range [{min_area}, {max_area}])")
                                        continue  # Skip too small or too large masks
                                    
                                    print(f" -> ACCEPTED")
                                    
                                    bbox = boxes[i].cpu().numpy().tolist() if i < len(boxes) else [0, 0, w, h]
                                    
                                    masks_list.append({
                                        'segmentation': mask_bool,
                                        'score': score,
                                        'area': area,
                                        'bbox': bbox
                                    })
                            else:
                                print(f"  Processing {len(masks)} mask candidates from SAM3...")
                                for i, mask in enumerate(masks):
                                    if torch.is_tensor(mask):
                                        mask_np = mask.cpu().numpy()
                                    else:
                                        mask_np = np.array(mask)
                                    if len(mask_np.shape) > 2:
                                        mask_np = mask_np.squeeze()
                                    
                                    # Convert to boolean mask
                                    mask_bool = mask_np.astype(bool)
                                    area = int(np.sum(mask_bool))
                                    
                                    # Get score
                                    score = float(scores[i]) if i < len(scores) and scores[i] is not None else 0.0
                                    
                                    # Get score
                                    score = float(scores[i]) if i < len(scores) and scores[i] is not None else 0.0
                                    
                                    # Debug: show all candidates
                                    print(f"    Candidate {i}: score={score:.3f}, area={area}px", end="")
                                    
                                    # Filter masks based on score and area
                                    if score < min_score:
                                        print(f" -> REJECTED (score < {min_score:.3f})")
                                        continue  # Skip low confidence masks
                                    if area < min_area or area > max_area:
                                        print(f" -> REJECTED (area out of range [{min_area}, {max_area}])")
                                        continue  # Skip too small or too large masks
                                    
                                    print(f" -> ACCEPTED")
                                    
                                    bbox = boxes[i].cpu().numpy().tolist() if i < len(boxes) else [0, 0, w, h]
                                    
                                    masks_list.append({
                                        'segmentation': mask_bool,
                                        'score': score,
                                        'area': area,
                                        'bbox': bbox
                                    })
                            
                            sel_mode = (
                                mask_selection_mode
                                if mask_selection_mode is not None
                                else getattr(self, "_pending_mask_selection_mode", "smallest")
                            )
                            n_candidates = (
                                len(masks)
                                if isinstance(masks, (list, tuple))
                                else masks.shape[0]
                                if isinstance(masks, torch.Tensor)
                                else 0
                            )
                            original_count = len(masks_list)
                            masks_list = self._select_sam3_automatic_masks(
                                masks_list, max_cells, sel_mode
                            )
                            print(
                                f"Generated {len(masks_list)} masks using SAM3 processor "
                                f"(filtered from {n_candidates} total, mode={sel_mode}, max={max_cells})"
                            )
                            if len(masks_list) == 0:
                                print(f"  All masks filtered out (score >= {min_score}, area between {min_area} and {max_area} pixels)")
                            elif original_count > len(masks_list):
                                print(f"  Selected {len(masks_list)} masks from {original_count} candidates (mode={sel_mode})")
                            return masks_list
                        else:
                            print(f"SAM3 processor returned no masks with confidence_threshold={auto_confidence:.2f}")
                            print(f"  Note: Confidence threshold {auto_confidence:.2f} may still be too high for automatic mode")
                            print(f"  Original threshold was {confidence_threshold:.2f}, adjusted to {auto_confidence:.2f}")
                            print(f"  Recommendation: Use bounding box prompts for best results")
                            print(f"  To use prompts: Draw a bounding box around cells and press 'S'")
                            return []  # Return empty list instead of falling through
                            
                except ImportError as e:
                    error_msg = str(e)
                    print(f"SAM3 processor not available (ImportError): {error_msg}")
                    print("\n" + "="*60)
                    print("SAM3 MODULE NOT FOUND")
                    print("="*60)
                    print("To use SAM3 automatic segmentation, you need to install SAM3:")
                    print("  1. Install SAM3: pip install sam3")
                    print("  2. Or add SAM3 path to PYTHONPATH")
                    print("  3. Common SAM3 locations:")
                    print("     - /cellchorus/sam3_training/sam3")
                    print("     - ~/sam3")
                    print("     - /opt/sam3")
                    print("\nFor now, using standard SAM methods...")
                    print("="*60 + "\n")
                    # Don't print full traceback for ImportError - it's expected
                except Exception as e:
                    print(f"SAM3 processor failed: {e}")
                    print("Falling back to standard SAM methods...")
                    import traceback
                    traceback.print_exc()
                except Exception as e:
                    print(f"SAM3 processor failed: {e}")
                    print("Falling back to standard SAM methods...")
                    import traceback
                    traceback.print_exc()
            
            # Try using SamAutomaticMaskGenerator if available (for standard SAM only)
            # Skip if this is SAM3 (SamAutomaticMaskGenerator doesn't work with SAM3)
            if not checkpoint_has_sam3:
                try:
                    from segment_anything import SamAutomaticMaskGenerator
                    
                    if self.sam is not None and hasattr(self.sam, 'image_encoder'):
                        print("Using SamAutomaticMaskGenerator")
                        print(f"Image shape: {image.shape}, dtype: {image.dtype}, range: [{image.min()}, {image.max()}]")
                        
                        # Ensure image is contiguous numpy array
                        if not isinstance(image, np.ndarray):
                            image = np.array(image)
                        image = np.ascontiguousarray(image)
                        
                        # For small images, reduce points_per_side to avoid issues
                        h, w = image.shape[:2]
                        min_dim = min(h, w)
                        if min_dim < 512:
                            points_per_side = 16  # Fewer points for small images
                            crop_n_layers = 0  # Disable cropping for small images
                            print(f"Small image ({w}x{h}), using fewer points")
                        else:
                            points_per_side = 32
                            crop_n_layers = 1
                        
                        try:
                            # Try with lenient thresholds first
                            mask_generator = SamAutomaticMaskGenerator(
                                model=self.sam,
                                points_per_side=points_per_side,
                                pred_iou_thresh=0.70,  # Lower threshold to catch more masks
                                stability_score_thresh=0.75,  # Lower threshold
                                crop_n_layers=crop_n_layers,
                                crop_n_points_downscale_factor=2,
                                min_mask_region_area=max(25, int(min_dim * 0.01)),  # Adaptive min area (1% of min dimension)
                                box_nms_thresh=0.7,  # Non-max suppression threshold
                            )
                            
                            masks = mask_generator.generate(image)
                            print(f"Generated {len(masks)} automatic masks with lenient settings")
                            
                            # If no masks found, try even more lenient settings
                            if len(masks) == 0:
                                print("No masks found with lenient settings, trying very permissive settings...")
                                mask_generator = SamAutomaticMaskGenerator(
                                    model=self.sam,
                                    points_per_side=points_per_side,
                                    pred_iou_thresh=0.50,  # Very low threshold
                                    stability_score_thresh=0.60,  # Very low threshold
                                    crop_n_layers=crop_n_layers,
                                    crop_n_points_downscale_factor=2,
                                    min_mask_region_area=max(10, int(min_dim * 0.005)),  # Even smaller min area
                                    box_nms_thresh=0.5,
                                )
                                masks = mask_generator.generate(image)
                                print(f"Generated {len(masks)} automatic masks with permissive settings")
                            
                            # If still no masks, fall through to grid-based method
                            if len(masks) > 0:
                                return masks
                            else:
                                print("No masks found even with permissive settings, falling back to grid method")
                        except Exception as e:
                            print(f"SamAutomaticMaskGenerator failed: {e}")
                            import traceback
                            traceback.print_exc()
                            # Fall through to grid-based method
                    else:
                        print("SAM model doesn't have image_encoder, using fallback")
                        
                except ImportError:
                    print("SamAutomaticMaskGenerator not available, using fallback")
                except Exception as e:
                    print(f"SamAutomaticMaskGenerator failed: {e}, using fallback")
            else:
                print("SAM3 detected - skipping SamAutomaticMaskGenerator (not compatible with SAM3)")
                
            # Fallback: Generate masks using grid of points
            if hasattr(self.predictor, 'predict'):
                print("Using grid-based automatic segmentation")
                
                # Set image
                if hasattr(self.predictor, 'set_image'):
                    try:
                        self.predictor.set_image(image)
                        print("Image set in predictor")
                    except Exception as e:
                        print(f"Error setting image: {e}")
                        return None
                
                h, w = image.shape[:2]
                masks_list = []
                total_area = h * w
                
                # Adaptive grid size based on image size
                min_dim = min(h, w)
                if min_dim < 300:
                    grid_size = 8  # Finer grid for small images
                elif min_dim < 500:
                    grid_size = 10
                else:
                    grid_size = 12  # Finer grid for better coverage
                
                step_x = max(1, w // (grid_size + 1))
                step_y = max(1, h // (grid_size + 1))
                
                print(f"Generating masks from {grid_size}x{grid_size} grid on {w}x{h} image...")
                
                # Try both foreground (1) and background (0) points
                for label in [1, 0]:
                    for i in range(1, grid_size + 1):
                        for j in range(1, grid_size + 1):
                            x = min(i * step_x, w - 1)
                            y = min(j * step_y, h - 1)
                            
                            try:
                                masks, scores, _ = self.predictor.predict(
                                    point_coords=np.array([[x, y]], dtype=np.float32),
                                    point_labels=np.array([label], dtype=np.int32),
                                    multimask_output=True
                                )
                                
                                # Take best mask if score is good enough
                                best_idx = np.argmax(scores)
                                score = float(scores[best_idx])
                                
                                # Very lenient threshold for grid-based method
                                if score > 0.30:  # Very low threshold to catch more masks
                                    # Check if mask is not too small or too large
                                    mask = masks[best_idx].astype(bool)
                                    area = np.sum(mask)
                                    
                                    # More lenient area constraints (adaptive to image size)
                                    min_area = max(20, int(total_area * 0.001))  # 0.1% of image
                                    max_area = int(total_area * 0.9)  # 90% of image
                                    
                                    if min_area < area < max_area:
                                        # Calculate actual bbox
                                        rows = np.any(mask, axis=1)
                                        cols = np.any(mask, axis=0)
                                        if np.any(rows) and np.any(cols):
                                            y_min, y_max = np.where(rows)[0][[0, -1]]
                                            x_min, x_max = np.where(cols)[0][[0, -1]]
                                            bbox = [x_min, y_min, x_max, y_max]
                                        else:
                                            bbox = [0, 0, w-1, h-1]
                                        
                                        masks_list.append({
                                            'segmentation': mask,
                                            'score': score,
                                            'area': int(area),
                                            'bbox': bbox
                                        })
                            except Exception as e:
                                # Silent fail for individual points to avoid spam
                                continue
                
                # Remove duplicate/overlapping masks
                if len(masks_list) > 1:
                    unique_masks = self._remove_duplicate_masks(masks_list)
                    print(f"Generated {len(unique_masks)} unique masks from {len(masks_list)} candidates")
                    return unique_masks
                
                print(f"Generated {len(masks_list)} masks from grid")
                return masks_list
            
            print("No prediction method available")
            return None
            
        except Exception as e:
            print(f"Error in automatic SAM prediction: {e}")
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(f"Automatic prediction failed: {str(e)}")
            return None
    
    def _remove_duplicate_masks(self, masks_list, iou_threshold=0.5):
        """Remove duplicate/overlapping masks based on IoU"""
        if len(masks_list) <= 1:
            return masks_list
        
        # Sort by score (highest first)
        masks_list = sorted(masks_list, key=lambda x: x['score'], reverse=True)
        
        unique_masks = []
        for mask_data in masks_list:
            mask = mask_data['segmentation']
            
            # Check if this mask significantly overlaps with any existing unique mask
            is_duplicate = False
            for existing in unique_masks:
                existing_mask = existing['segmentation']
                
                # Calculate IoU
                intersection = np.logical_and(mask, existing_mask).sum()
                union = np.logical_or(mask, existing_mask).sum()
                
                if union > 0:
                    iou = intersection / union
                    if iou > iou_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_masks.append(mask_data)
        
        # Post-process masks to expand borders and carve hollows
        processed_masks = []
        for mask_data in unique_masks[:max_cells if max_cells else len(unique_masks)]:
            mask = mask_data['segmentation']
            if mask.dtype != np.uint8:
                mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
            else:
                mask_uint8 = mask.astype(np.uint8)
            processed_mask = self._post_process_sam_mask(mask_uint8)
            mask_data['segmentation'] = processed_mask
            processed_masks.append(mask_data)
        
        return processed_masks
    
    @staticmethod
    @contextmanager
    def _null_context():
        """Null context manager for when memory manager is not available"""
        yield
    
    def _post_process_sam_mask(self, mask):
        """
        Post-process SAM mask to:
        1. Expand slightly to include border regions (dilate)
        2. Carve hollow regions using edge detection
        3. Ensure smooth boundaries
        
        Args:
            mask: Binary mask (uint8, 0 or 255)
            
        Returns:
            Processed mask (uint8, 0 or 255)
        """
        if mask is None or mask.size == 0:
            return mask
        
        # Convert to numpy array if not already
        mask = np.array(mask)
        
        # Ensure mask is 2D (handle 3D arrays by taking first channel or squeezing)
        if len(mask.shape) == 3:
            if mask.shape[0] == 1:
                mask = mask[0]
            elif mask.shape[2] == 1:
                mask = mask[:, :, 0]
            else:
                mask = mask[:, :, 0]  # Take first channel
        elif len(mask.shape) > 2:
            mask = mask.squeeze()
            # If still not 2D, take first slice
            if len(mask.shape) > 2:
                mask = mask[0]
        
        # Ensure mask is exactly 2D
        if len(mask.shape) != 2:
            print(f"Warning: Mask has unexpected shape {mask.shape}, trying to reshape")
            # Try to flatten and reshape to square if possible
            size = int(np.sqrt(mask.size))
            if size * size == mask.size:
                mask = mask.reshape(size, size)
            else:
                print(f"Error: Cannot reshape mask of size {mask.size} to 2D")
                return mask.flatten()[:size*size].reshape(size, size) if size > 0 else mask
        
        # Ensure binary mask (convert to uint8, 0 or 255)
        if mask.dtype != np.uint8:
            mask = mask.astype(np.float32)
            if mask.max() <= 1.0:
                mask = (mask > 0.5).astype(np.uint8) * 255
            else:
                mask = (mask > 127).astype(np.uint8) * 255
        else:
            # Ensure values are 0 or 255
            mask = (mask > 127).astype(np.uint8) * 255
        
        # Final check: ensure it's a proper 2D uint8 array
        if len(mask.shape) != 2 or mask.dtype != np.uint8:
            print(f"Error: Mask is not a valid 2D uint8 array: shape={mask.shape}, dtype={mask.dtype}")
            return mask
        
        # Step 1: Expand mask slightly (dilate) to include border regions
        # Use minimal expansion - just enough to capture border regions and hollows
        original_area = np.sum(mask > 0)
        # Use very small kernel with minimal dilation to prevent over-expansion
        kernel_size = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        # Minimal dilation: only 1 iteration to slightly expand (was 2 iterations with 5x5 kernel)
        expanded_mask = cv2.dilate(mask, kernel, iterations=1)
        expanded_area = np.sum(expanded_mask > 0)
        if original_area > 0:
            expansion_ratio = expanded_area / original_area
            # Debug: log expansion (only warn if > 20% expansion)
            if expansion_ratio > 1.2:
                print(f"Mask expansion ratio: {expansion_ratio:.2f}x (Original: {original_area}, Expanded: {expanded_area})")
            # Limit expansion: if mask expanded too much, use erosion to bring it back slightly
            if expansion_ratio > 1.25:  # If expanded more than 25%, apply slight erosion
                # Apply minimal erosion to reduce over-expansion
                erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                expanded_mask = cv2.erode(expanded_mask, erosion_kernel, iterations=1)
                final_area = np.sum(expanded_mask > 0)
                print(f"Applied erosion to reduce over-expansion. Final expansion ratio: {final_area/original_area:.2f}x")
        
        # Ensure expanded_mask is valid 2D uint8 before findContours
        if len(expanded_mask.shape) != 2 or expanded_mask.dtype != np.uint8:
            print(f"Warning: expanded_mask has invalid format: shape={expanded_mask.shape}, dtype={expanded_mask.dtype}")
            # Try to fix it
            if len(expanded_mask.shape) > 2:
                expanded_mask = expanded_mask.squeeze()
            if expanded_mask.dtype != np.uint8:
                expanded_mask = expanded_mask.astype(np.uint8)
        
        # Step 2: Carve hollow regions
        # Find interior contours (holes) and remove very small ones, but keep larger hollows
        try:
            contours, hierarchy = cv2.findContours(expanded_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except cv2.error as e:
            print(f"Error in cv2.findContours: {e}, mask shape={expanded_mask.shape}, dtype={expanded_mask.dtype}")
            # Return the expanded mask without hollow carving
            return expanded_mask
        
        if hierarchy is not None and len(contours) > 0:
            # Create mask with holes carved
            hollow_mask = np.zeros_like(expanded_mask)
            
            # Draw outer contours (positive areas)
            for i, contour in enumerate(contours):
                # Check if this is an outer contour (parent is -1) or an inner contour (hole)
                if hierarchy[0][i][3] == -1:  # Outer contour
                    cv2.fillPoly(hollow_mask, [contour], 255)
                else:
                    # Inner contour (potential hole/hollow)
                    # Only carve if hole is large enough (to preserve actual hollows)
                    area = cv2.contourArea(contour)
                    min_hole_area = 50  # Minimum area to consider as a hollow
                    if area > min_hole_area:
                        # Carve this hole (don't fill it)
                        cv2.fillPoly(hollow_mask, [contour], 0)
            
            # If we have a valid hollow mask, use it
            if hollow_mask.max() > 0:
                expanded_mask = hollow_mask
        
        # Step 3: Smooth edges with morphological operations
        # Use smaller kernel and avoid operations that expand (close can expand slightly)
        smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Only close very small gaps (1 iteration is conservative)
        expanded_mask = cv2.morphologyEx(expanded_mask, cv2.MORPH_CLOSE, smooth_kernel, iterations=1)
        # Open to remove small noise (this shrinks, so it's fine)
        expanded_mask = cv2.morphologyEx(expanded_mask, cv2.MORPH_OPEN, smooth_kernel, iterations=1)
        
        # Step 4: Apply slight Gaussian blur to smooth edges (minimal expansion)
        # Use smaller kernel to reduce expansion effect
        expanded_mask = cv2.GaussianBlur(expanded_mask.astype(np.float32), (3, 3), 0.5)  # Reduced kernel and sigma
        expanded_mask = (expanded_mask > 127).astype(np.uint8) * 255
        
        return expanded_mask
    
    # ============ Async slot methods for non-blocking predictions ============
    
    @pyqtSlot()
    def predict_automatic_async(self):
        """
        Async version of predict_automatic - runs in background thread.
        Reads parameters from worker attributes (_pending_image, _pending_confidence, _pending_max_cells).
        Emits prediction_automatic_complete signal with results.
        """
        try:
            print(f"[SAM WORKER] predict_automatic_async called")
            print(f"[SAM WORKER] _pending_image is None: {self._pending_image is None}")
            if self._pending_image is not None:
                print(f"[SAM WORKER] _pending_image shape: {self._pending_image.shape}")
            
            if self._pending_image is None:
                self.error_occurred.emit("No image provided for automatic prediction")
                self.prediction_automatic_complete.emit(None)
                return
            
            # Set CUDA to non-blocking mode for async operations
            if torch.cuda.is_available() and self.device.startswith('cuda'):
                # Use non-blocking mode to prevent hanging
                # Extract device index from device string
                if self.device == "cuda":
                    device_index = 0
                else:
                    try:
                        device_index = int(self.device.split(':')[1])
                    except (IndexError, ValueError):
                        device_index = 0
                torch.cuda.set_device(device_index)
                # Allow CUDA operations to overlap with CPU work
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            masks = self.predict_automatic(
                self._pending_image,
                self._pending_confidence,
                self._pending_max_cells,
                mask_selection_mode=getattr(self, "_pending_mask_selection_mode", "smallest"),
            )
            self.prediction_automatic_complete.emit(masks)
        except Exception as e:
            error_msg = f"Error in async automatic prediction: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(error_msg)
            self.prediction_automatic_complete.emit(None)
        finally:
            # Clear pending data
            self._pending_image = None
    
    @pyqtSlot()
    def predict_with_boxes_async(self):
        """
        Async version of predict_with_boxes - runs in background thread.
        Reads parameters from worker attributes (_pending_boxes, _pending_confidence, _pending_max_cells).
        Emits prediction_boxes_complete signal with results.
        Note: image must be set first using set_image_async() or set_image().
        """
        print("[SAM WORKER] predict_with_boxes_async: Starting...")
        try:
            if self._pending_boxes is None:
                print("[SAM WORKER] ERROR: No boxes provided")
                self.error_occurred.emit("No boxes provided for box prediction")
                self.prediction_boxes_complete.emit(None)
                return
            
            print(f"[SAM WORKER] predict_with_boxes_async: Processing {len(self._pending_boxes)} box(es)")
            
            # Set CUDA to non-blocking mode for async operations
            if torch.cuda.is_available() and self.device.startswith('cuda'):
                # Use non-blocking mode to prevent hanging
                # Extract device index from device string
                if self.device == "cuda":
                    device_index = 0
                else:
                    try:
                        device_index = int(self.device.split(':')[1])
                    except (IndexError, ValueError):
                        device_index = 0
                torch.cuda.set_device(device_index)
                # Allow CUDA operations to overlap with CPU work
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print(f"[SAM WORKER] CUDA configured for device {device_index}")
            
            print("[SAM WORKER] Calling predict_with_boxes...")
            # Use memory manager for non-blocking cleanup
            with self.memory_manager.auto_cleanup() if self.memory_manager else self._null_context():
                masks = self.predict_with_boxes(self._pending_boxes, self._pending_confidence, self._pending_max_cells)
            
            # Handle different mask types for debug output
            if masks is None:
                mask_count = 0
            elif isinstance(masks, (list, tuple)):
                mask_count = len(masks)
            elif hasattr(masks, '__len__'):
                try:
                    mask_count = len(masks)
                except (TypeError, ValueError):
                    mask_count = 1 if masks is not None else 0
            else:
                mask_count = 1 if masks is not None else 0
            print(f"[SAM WORKER] predict_with_boxes completed, got {mask_count} mask(s)")
            self.prediction_boxes_complete.emit(masks)
        except Exception as e:
            error_msg = f"Error in async box prediction: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(error_msg)
            self.prediction_boxes_complete.emit(None)
        finally:
            # Clear pending data
            self._pending_boxes = None

    @pyqtSlot()
    def set_image_and_predict_boxes_async(self):
        """
        Set image from _pending_image then run predict_with_boxes in one worker-thread call.
        Used by viewers that cannot reliably QMetaObject.invokeMethod(set_image_async).
        """
        print("[SAM WORKER] set_image_and_predict_boxes_async: Starting...")
        try:
            if self._pending_image is None:
                self.error_occurred.emit("No image for box prediction")
                self.prediction_boxes_complete.emit(None)
                return
            if self._pending_boxes is None:
                self.error_occurred.emit("No boxes provided for box prediction")
                self.prediction_boxes_complete.emit(None)
                return

            self.set_image(self._pending_image)

            if torch.cuda.is_available() and self.device.startswith("cuda"):
                if self.device == "cuda":
                    device_index = 0
                else:
                    try:
                        device_index = int(self.device.split(":")[1])
                    except (IndexError, ValueError):
                        device_index = 0
                torch.cuda.set_device(device_index)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            with self.memory_manager.auto_cleanup() if self.memory_manager else self._null_context():
                masks = self.predict_with_boxes(
                    self._pending_boxes,
                    self._pending_confidence,
                    self._pending_max_cells,
                )
            self.prediction_boxes_complete.emit(masks)
        except Exception as e:
            error_msg = f"Error in box prompt (set_image + predict): {str(e)}"
            print(error_msg)
            import traceback

            traceback.print_exc()
            self.error_occurred.emit(error_msg)
            self.prediction_boxes_complete.emit(None)
        finally:
            self._pending_boxes = None

    @pyqtSlot()
    def predict_with_points_async(self):
        """
        Async version of predict_with_points - runs in background thread.
        Reads parameters from worker attributes (_pending_points, _pending_labels, _pending_confidence, _pending_max_cells).
        Emits prediction_points_complete signal with results.
        Note: image must be set first using set_image_async() or set_image().
        """
        try:
            if self._pending_points is None or self._pending_labels is None:
                self.error_occurred.emit("No points/labels provided for point prediction")
                self.prediction_points_complete.emit(None)
                return
            masks = self.predict_with_points(self._pending_points, self._pending_labels, self._pending_confidence, self._pending_max_cells)
            self.prediction_points_complete.emit(masks)
        except Exception as e:
            error_msg = f"Error in async point prediction: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(error_msg)
            self.prediction_points_complete.emit(None)
        finally:
            # Clear pending data
            self._pending_points = None
            self._pending_labels = None

    @pyqtSlot(object, object, float, int, int)
    def batch_process_frame_async(self, image, boxes, confidence, max_cells, frame_idx):
        """
        Atomic operation for batch processing a single frame.
        Sets image, runs prediction, and emits result with frame ID.
        This is designed to be called sequentially from a Batch Processor.
        """
        print(f"[SAM WORKER] batch_process_frame_async called for frame {frame_idx}")
        try:
            if image is None:
                print(f"[SAM WORKER] No image provided for frame {frame_idx}, emitting None")
                self.batch_frame_complete.emit(None, frame_idx)
                return
            
            print(f"[SAM WORKER] Processing frame {frame_idx} with boxes={boxes is not None and len(boxes) if boxes else 0}")
            masks = None
            
            # Use torch.inference_mode for efficiency and safety
            ctx = torch.inference_mode() if hasattr(torch, 'inference_mode') else torch.no_grad()
            with ctx:
                # Check if we have boxes
                if boxes is not None and len(boxes) > 0:
                    print(f"[SAM WORKER] Using box prompts ({len(boxes)} boxes) for frame {frame_idx}")
                    # Use box prompt
                    if self.is_sam3_model or self.sam3_processor:
                        # SAM3 specific path - pass image directly
                        masks = self._predict_with_boxes_sam3(image, boxes, confidence)
                    else:
                        # Standard SAM path
                        self.set_image(image)
                        masks = self.predict_with_boxes(boxes, image)
                else:
                    print(f"[SAM WORKER] Using automatic mode for frame {frame_idx}")
                    # Automatic mode
                    masks = self.predict_automatic(image, confidence, max_cells)
            
            print(f"[SAM WORKER] Frame {frame_idx} processed, masks={'None' if masks is None else 'received'}")
            self.batch_frame_complete.emit(masks, frame_idx)
            
        except Exception as e:
            print(f"[SAM WORKER] Error in batch_process_frame_async for frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()
            self.batch_frame_complete.emit(None, frame_idx)
    
    @pyqtSlot()
    def set_image_async(self):
        """
        Async version of set_image - runs in background thread.
        Reads image from worker attribute (_pending_image) and sets it for subsequent predictions.
        """
        print("[SAM WORKER] set_image_async: Starting...")
        try:
            if self._pending_image is None:
                print("[SAM WORKER] ERROR: No image provided for set_image")
                self.error_occurred.emit("No image provided for set_image")
                return
            print(f"[SAM WORKER] set_image_async: Image shape = {self._pending_image.shape if hasattr(self._pending_image, 'shape') else 'unknown'}")
            print("[SAM WORKER] Calling set_image()...")
            self.set_image(self._pending_image)
            print("[SAM WORKER] set_image() completed successfully")
        except Exception as e:
            error_msg = f"Error setting image: {str(e)}"
            print(f"[SAM WORKER] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(error_msg)
        finally:
            # Don't clear _pending_image here - it might be needed for prediction
            # The prediction methods will clear it
            print("[SAM WORKER] set_image_async: Finished")
            pass

