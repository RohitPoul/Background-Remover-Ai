"""
Safe wrapper for CUDA operations to prevent crashes
"""
import torch
import gc

def safe_cuda_empty_cache():
    """Safely empty CUDA cache without crashing on errors"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[WARNING] CUDA out of memory during cache clear - GPU may need restart")
            # Try to recover by collecting garbage first
            gc.collect()
            try:
                # Try once more after garbage collection
                torch.cuda.empty_cache()
                return True
            except:
                print(f"[WARNING] Unable to clear CUDA cache - continuing anyway")
                return False
        else:
            print(f"[WARNING] CUDA error during cache clear: {e}")
            return False
    except Exception as e:
        print(f"[WARNING] Unexpected error clearing CUDA cache: {e}")
        return False

def safe_cuda_synchronize():
    """Safely synchronize CUDA without crashing on errors"""
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return True
    except RuntimeError as e:
        print(f"[WARNING] CUDA error during synchronize: {e}")
        return False
    except Exception as e:
        print(f"[WARNING] Unexpected error during CUDA synchronize: {e}")
        return False
