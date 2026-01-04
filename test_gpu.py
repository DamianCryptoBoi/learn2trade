import torch
import sys

def test_cuda():
    print("-" * 30)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        
        # Test a simple tensor operation on GPU
        try:
            x = torch.rand(3, 3).to("cuda")
            y = torch.rand(3, 3).to("cuda")
            z = x @ y
            print("Successfully performed matrix multiplication on GPU!")
        except Exception as e:
            print(f"Error during GPU operation: {e}")
    else:
        print("\n[!] CUDA is NOT available.")
        if "cpu" in torch.__version__:
            print("Reason: You have the '+cpu' version of PyTorch installed.")
        else:
            print("Reason: PyTorch cannot find your NVIDIA drivers or compatible hardware.")
    
    print("-" * 30)

if __name__ == "__main__":
    test_cuda()
