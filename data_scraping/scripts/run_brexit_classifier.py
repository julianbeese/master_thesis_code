#!/usr/bin/env python3
"""
Brexit Classifier Runner
Hauptskript zum Ausführen der Brexit-Debatten-Klassifikation
"""

import sys
from pathlib import Path

# Füge Projekt-Root zum Python-Pfad hinzu
sys.path.append(str(Path(__file__).parent.parent))

# Importiere aus dem classifier Package
from classifier.run_brexit_filtering import main

if __name__ == "__main__":
    main()
