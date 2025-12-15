# Parliament Data Database System

Ein umfassendes Datenbank-Management-System für XML-Scraping und -Parsing von Parlamentsdaten.

## Tech-Stack

- **DuckDB**: Hauptdatenbank für analytische Workloads
- **Polars**: Hochperformante DataFrame-Bibliothek
- **Pandas**: Traditionelle Datenanalyse
- **PyArrow**: Effiziente Speicherung und Übertragung von Daten

## Struktur

```
database/
├── config/                 # Konfigurationsdateien
│   ├── __init__.py
│   └── database_config.py
├── managers/               # Datenbank-Manager
│   ├── __init__.py
│   └── database_manager.py
├── utils/                  # Utility-Funktionen
│   ├── __init__.py
│   └── data_utils.py
├── data/                   # Datenverzeichnisse
│   ├── raw/               # Rohe XML-Daten
│   ├── processed/         # Verarbeitete Daten
│   └── backups/           # Datenbank-Backups
├── logs/                  # Log-Dateien
└── __init__.py
```

## Features

### DatabaseManager
- DuckDB-Integration mit automatischer Verbindungsverwaltung
- Unterstützung für Polars, Pandas und PyArrow
- Parquet-Import/Export
- Tabellen-Management und -Informationen
- Context Manager für sichere Verbindungen

### DataConverter
- Nahtlose Konvertierung zwischen Polars, Pandas und PyArrow
- Optimierte Performance für große Datensätze

### DataValidator
- Umfassende Datenvalidierung
- Schema-Überprüfung
- Null-Werte-Detection
- Benutzerdefinierte Validierungsregeln

### DataProcessor
- Datenbereinigung und -transformation
- Batch-Verarbeitung für große Datensätze
- Duplikat-Entfernung
- String-Bereinigung

### FileManager
- Multi-Format-Unterstützung (Parquet, CSV, JSON)
- Automatische Verzeichniserstellung
- Datei-Listing und -Suche

### PerformanceMonitor
- Performance-Tracking
- Memory-Monitoring
- Detaillierte Logging-Informationen

## Installation

```bash
pip install -r requirements.txt
```

## Verwendung

### Basis-Setup

```python
from database import DatabaseManager, config

# Konfiguration initialisieren
config.setup_directories()

# Datenbank-Manager erstellen
with DatabaseManager() as db:
    # Datenbank-Operationen
    pass
```

### Datenverarbeitung

```python
import polars as pl
from database import DatabaseManager, DataConverter

# Polars DataFrame erstellen
df = pl.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})

# In Datenbank speichern
with DatabaseManager() as db:
    db.create_table_from_polars(df, "users")
    
    # Als Parquet exportieren
    db.save_to_parquet("users", "data/processed/users.parquet")
```

### Datenvalidierung

```python
from database import DataValidator, validate_and_clean

# Validierung
validator = DataValidator()
result = validator.validate_dataframe(df, required_columns=["name", "age"])

# Validierung und Bereinigung
clean_df = validate_and_clean(df, required_columns=["name", "age"])
```

## Konfiguration

Die Konfiguration erfolgt über `database/config/database_config.py`:

- Datenbank-Pfade
- Performance-Einstellungen
- Logging-Konfiguration
- XML-Parsing-Einstellungen

Umgebungsvariablen werden über `EnvironmentConfig` unterstützt.

## Logging

Automatisches Logging in `database/logs/` mit:
- Tägliche Log-Rotation
- Konfigurierbare Log-Level
- Performance-Metriken

## Nächste Schritte

1. **Models erstellen**: Definieren Sie Ihre Datenmodelle basierend auf den XML-Strukturen
2. **XML-Parser implementieren**: Nutzen Sie die Utility-Funktionen für XML-Verarbeitung
3. **Scraping-Pipeline**: Implementieren Sie den XML-Download und -Verarbeitungsprozess
4. **Datenanalyse**: Nutzen Sie die integrierten Tools für Datenexploration und -analyse
