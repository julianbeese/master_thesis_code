#!/usr/bin/env python3
"""
GPU Test Script für RunPod
Testet ob CUDA korrekt funktioniert und das Model auf GPU läuft
"""

import torch
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def test_gpu():
    """Testet GPU-Funktionalität"""
    print("=" * 50)
    print("GPU TEST SCRIPT")
    print("=" * 50)
    
    # 1. CUDA Verfügbarkeit testen
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA ist NICHT verfügbar!")
        print("Das Training wird auf CPU laufen (sehr langsam)")
        return False
    
    # 2. GPU Details anzeigen
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  - Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  - Compute Capability: {props.major}.{props.minor}")
    
    # 3. Einfacher GPU-Test
    print("\n" + "=" * 30)
    print("GPU COMPUTATION TEST")
    print("=" * 30)
    
    try:
        device = torch.device("cuda:0")
        print(f"Using device: {device}")
        
        # Einfache Matrix-Operation
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        c = torch.matmul(a, b)
        
        print("✅ GPU Matrix multiplication successful")
        print(f"Result shape: {c.shape}")
        print(f"Result device: {c.device}")
        
    except Exception as e:
        print(f"❌ GPU computation failed: {e}")
        return False
    
    # 4. Model Loading Test
    print("\n" + "=" * 30)
    print("MODEL LOADING TEST")
    print("=" * 30)
    
    try:
        model_name = "mistralai/Mistral-7B-v0.1"
        print(f"Loading model: {model_name}")
        
        # Tokenizer laden
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Model laden (kleine Version für Test)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=6,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            load_in_4bit=True
        )
        
        print("✅ Model loaded successfully")
        print(f"Model device: {next(model.parameters()).device}")
        
        # Test inference
        test_text = "This is a test sentence for GPU inference."
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        # Move inputs to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print("✅ Model inference successful")
        print(f"Output shape: {outputs.logits.shape}")
        print(f"Output device: {outputs.logits.device}")
        
        # Memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"GPU Memory Allocated: {memory_allocated:.2f} GB")
            print(f"GPU Memory Reserved: {memory_reserved:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def main():
    """Hauptfunktion"""
    success = test_gpu()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("GPU ist bereit für Training!")
    else:
        print("❌ TESTS FAILED!")
        print("GPU ist NICHT bereit für Training!")
    print("=" * 50)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
