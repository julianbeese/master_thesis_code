#!/bin/bash
# Quick Start Script für beide Training-Pipelines

set -e  # Exit on error

echo "======================================"
echo "Master Thesis ML Training - Quick Start"
echo "======================================"
echo ""

# Farben für Output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Funktion zum Anzeigen von Optionen
show_menu() {
    echo "${BLUE}Wähle eine Training-Pipeline:${NC}"
    echo ""
    echo "1) Classification Pipeline"
    echo "   - Schnelles Training (3 epochs)"
    echo "   - Multi-Class Classification"
    echo "   - Learning Rate: 2e-4"
    echo ""
    echo "2) Supervised Fine-Tuning Pipeline"
    echo "   - Tiefes Fine-Tuning (5 epochs)"
    echo "   - Bessere Generalisierung"
    echo "   - Learning Rate: 5e-5"
    echo ""
    echo "3) Beide Pipelines nacheinander"
    echo ""
    echo "4) Abbrechen"
    echo ""
}

# Funktion zum Prüfen der Daten
check_data() {
    echo "${BLUE}Prüfe Training-Daten...${NC}"
    
    if [ ! -f "data/training_data_export.csv" ] || [ ! -f "data/test_data_export.csv" ]; then
        echo "${RED}✗ Training-Daten nicht gefunden!${NC}"
        echo "${YELLOW}Bitte stelle sicher, dass folgende Dateien existieren:${NC}"
        echo "  - data/training_data_export.csv"
        echo "  - data/test_data_export.csv"
        exit 1
    fi
    
    echo "${GREEN}✓ Training-Daten gefunden${NC}"
}

# Funktion zum Prüfen der Dependencies
check_dependencies() {
    echo "${BLUE}Prüfe Python-Dependencies...${NC}"
    
    if ! command -v python3 &> /dev/null; then
        echo "${RED}✗ Python3 nicht gefunden!${NC}"
        exit 1
    fi
    
    echo "${GREEN}✓ Python3 gefunden${NC}"
}

# Classification Pipeline trainieren
run_classification() {
    echo ""
    echo "${GREEN}======================================"
    echo "Starte Classification Pipeline"
    echo "======================================${NC}"
    echo ""
    
    cd classification/src
    python train.py
    cd ../..
    
    echo ""
    echo "${GREEN}✓ Classification Training abgeschlossen!${NC}"
    echo "${YELLOW}Checkpoints gespeichert in: classification/checkpoints/${NC}"
}

# Supervised Fine-Tuning Pipeline trainieren
run_supervised_finetuning() {
    echo ""
    echo "${GREEN}======================================"
    echo "Starte Supervised Fine-Tuning Pipeline"
    echo "======================================${NC}"
    echo ""
    
    cd supervised_finetuning/src
    python train.py
    cd ../..
    
    echo ""
    echo "${GREEN}✓ Supervised Fine-Tuning abgeschlossen!${NC}"
    echo "${YELLOW}Checkpoints gespeichert in: supervised_finetuning/checkpoints/${NC}"
}

# Hauptprogramm
main() {
    # Prüfungen
    check_dependencies
    check_data
    
    echo ""
    show_menu
    
    read -p "Deine Wahl (1-4): " choice
    
    case $choice in
        1)
            run_classification
            ;;
        2)
            run_supervised_finetuning
            ;;
        3)
            run_classification
            echo ""
            echo "${BLUE}Warte 5 Sekunden vor dem nächsten Training...${NC}"
            sleep 5
            run_supervised_finetuning
            ;;
        4)
            echo "${YELLOW}Training abgebrochen.${NC}"
            exit 0
            ;;
        *)
            echo "${RED}Ungültige Wahl!${NC}"
            exit 1
            ;;
    esac
    
    echo ""
    echo "${GREEN}======================================"
    echo "Alle Trainings abgeschlossen!"
    echo "======================================${NC}"
    echo ""
    echo "${YELLOW}Nächste Schritte:${NC}"
    echo "1. Prüfe die Metriken in Weights & Biases"
    echo "2. Evaluiere die Models"
    echo "3. Nutze die Models für Inference"
}

# Script ausführen
main
