#!/bin/bash
# RunPod Setup Script für GPT-OSS-20B Training auf H100

set -e

echo "================================================"
echo "RunPod H100 Setup für GPT-OSS-20B Fine-Tuning"
echo "================================================"

# System info anzeigen
echo -e "\n=== System Information ==="
nvidia-smi
echo ""
echo "CUDA Version: $(nvcc --version | grep release | awk '{print $6}')"
echo "Python Version: $(python3 --version)"
echo "PyTorch Version: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA Available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU Count: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
echo "GPU Name: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"

# Verzeichnis-Struktur erstellen
echo -e "\n=== Erstelle Verzeichnisse ==="
mkdir -p outputs checkpoints logs data

# Prüfe ob DuckDB-Datei existiert
echo -e "\n=== Prüfe Daten ==="
if [ -f "data/debates_brexit_chunked.duckdb" ]; then
    echo "✓ DuckDB-Datei gefunden"
    echo "  Größe: $(du -h data/debates_brexit_chunked.duckdb | cut -f1)"
else
    echo "⚠ WARNUNG: DuckDB-Datei nicht gefunden!"
    echo "  Bitte lade 'debates_brexit_chunked.duckdb' in den 'data/' Ordner hoch"
    exit 1
fi

# Weights & Biases Login (optional)
echo -e "\n=== Weights & Biases Setup ==="
if [ -z "$WANDB_API_KEY" ]; then
    echo "⚠ WANDB_API_KEY nicht gesetzt"
    echo "  Setze die Variable mit: export WANDB_API_KEY=your_key"
    echo "  Oder logge dich manuell ein mit: wandb login"
else
    echo "✓ WANDB_API_KEY gefunden"
    wandb login $WANDB_API_KEY
fi

# GPU Memory Check
echo -e "\n=== GPU Memory Check ==="
python3 -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB')
"

echo -e "\n=== Setup abgeschlossen ==="
echo ""
echo "Zum Starten des Trainings:"
echo "  python3 src/train.py"
echo ""
echo "Zum Monitoring:"
echo "  watch -n 1 nvidia-smi"
echo "  tensorboard --logdir outputs"
echo ""
