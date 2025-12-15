#!/bin/bash
# Script zum Erstellen eines separaten Git Repos für ml_training

echo "🚀 Erstelle separates Git Repository für ml_training"
echo ""

# Initialisiere Git Repo
git init

# Erstelle .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/
dist/
build/

# Training outputs
outputs/
checkpoints/
evaluation_results/
logs/
*.log

# Data - DuckDB wird NICHT committed (zu groß)
data/*.duckdb
data/*.db

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Weights & Biases
wandb/

# Model cache
models/
.cache/
EOF

echo "✅ .gitignore erstellt"

# Erstelle README für GitHub
cat > README_GITHUB.md << 'EOF'
# GPT-OSS-20B Fine-Tuning für Brexit Debate Classification

Fine-Tuning von GPT-OSS-20B (20B Parameter Model) für Multi-Class Text Classification auf Brexit Debate Daten.

## 🚀 Quick Start auf RunPod

### 1. Repository klonen
```bash
cd /workspace
git clone [DEIN_REPO_URL]
cd [REPO_NAME]
```

### 2. Datenbank hochladen
Lade deine `debates_brexit_chunked.duckdb` Datei in den `data/` Ordner hoch.

### 3. Setup
```bash
pip install -r requirements.txt
chmod +x runpod_setup.sh
./runpod_setup.sh
```

### 4. Training starten
```bash
# Optional: W&B Login
export WANDB_API_KEY=your_key

# Training starten
python3 src/train.py
```

## 📊 Model Details

- **Base Model**: openai-community/gpt-oss-20b
- **Method**: LoRA Fine-Tuning (0.4% trainable params)
- **Task**: Multi-Class Classification (6 classes)
- **GPU**: H100 SXM
- **Training Time**: ~3-6 hours

## 📁 Struktur

```
├── config/
│   └── training_config.yaml    # Hyperparameter
├── src/
│   ├── data_loader.py          # DuckDB Loader
│   ├── train.py                # Training Script
│   ├── inference.py            # Inference
│   └── evaluate.py             # Evaluation
├── data/                       # Datenbank hier hochladen
├── requirements.txt
├── Dockerfile
└── runpod_setup.sh
```

## 📖 Dokumentation

Siehe `RUNPOD_GUIDE.md` für detaillierte Anleitung.

## 🎯 Nach dem Training

```bash
# Evaluation
python3 src/evaluate.py

# Inference
python3 src/inference.py
```

## 📝 Lizenz

Beachte die Lizenzbedingungen von GPT-OSS-20B.
EOF

echo "✅ README_GITHUB.md erstellt"

# Git add
git add .

# Initial commit
git commit -m "Initial commit: GPT-OSS-20B fine-tuning setup"

echo ""
echo "✅ Git Repository initialisiert!"
echo ""
echo "📌 Nächste Schritte:"
echo "1. Erstelle ein neues Repository auf GitHub (https://github.com/new)"
echo "2. Nenne es z.B. 'gpt-oss-20b-brexit-finetuning'"
echo "3. Führe dann aus:"
echo ""
echo "   git remote add origin https://github.com/DEIN_USERNAME/DEIN_REPO_NAME.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "4. Dann auf RunPod:"
echo "   cd /workspace"
echo "   git clone https://github.com/DEIN_USERNAME/DEIN_REPO_NAME.git"
echo ""
