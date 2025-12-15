#!/usr/bin/env python3
"""
Test-Skript für die Brexit Speech Klassifikation
Testet mit 500 Speeches und macht eine Hochrechnung für alle 40.000
"""

import duckdb
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
import statistics
from typing import Dict, List, Tuple
from collections import defaultdict
import sys
import os
import subprocess

# Konfiguration
TEST_SIZE = 300  # Anzahl Speeches für Test
INPUT_DB = "/Users/julianbeese/Developer/Master/data_collection/data/processed/brexit_analysis.duckdb"
# Nutze direkt die Hauptdatenbank für den Test
TEST_DB = INPUT_DB  # Gleiche DB für Test
TOTAL_SPEECHES_TARGET = 40000  # Ziel für Hochrechnung

# ANSI Color Codes für schöne Ausgabe
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """Druckt einen formatierten Header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")

def print_section(text):
    """Druckt eine Section-Überschrift"""
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'-'*len(text)}{Colors.ENDC}")

def format_duration(seconds):
    """Formatiert Sekunden in lesbare Zeit"""
    if seconds < 60:
        return f"{seconds:.1f} Sekunden"
    elif seconds < 3600:
        return f"{seconds/60:.1f} Minuten"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}min"

def format_currency(amount):
    """Formatiert Währungsbeträge"""
    return f"${amount:.2f}" if amount >= 0.01 else f"${amount:.4f}"

def check_environment():
    """Überprüft ob alle Voraussetzungen erfüllt sind"""
    print_section("🔍 Umgebungsprüfung")
    
    checks_passed = True
    
    # Prüfe Input-Datenbank
    if Path(INPUT_DB).exists():
        print(f"{Colors.OKGREEN}✓{Colors.ENDC} Input-Datenbank gefunden: {INPUT_DB}")
    else:
        print(f"{Colors.FAIL}✗{Colors.ENDC} Input-Datenbank nicht gefunden: {INPUT_DB}")
        checks_passed = False
    
    # Prüfe .env Datei
    env_path = Path("/Users/julianbeese/Developer/Master/data_collection/.env")
    if env_path.exists():
        print(f"{Colors.OKGREEN}✓{Colors.ENDC} .env Datei gefunden")
        
        # Prüfe API Keys
        api_keys_found = []
        with open(env_path, 'r') as f:
            content = f.read()
            for key in ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", 
                       "DEEPSEEK_API_KEY", "MISTRAL_API_KEY", "TOGETHER_API_KEY"]:
                if key in content:
                    api_keys_found.append(key.replace("_API_KEY", ""))
        
        if api_keys_found:
            print(f"{Colors.OKGREEN}✓{Colors.ENDC} API Keys gefunden für: {', '.join(api_keys_found)}")
        else:
            print(f"{Colors.FAIL}✗{Colors.ENDC} Keine API Keys gefunden")
            checks_passed = False
    else:
        print(f"{Colors.FAIL}✗{Colors.ENDC} .env Datei nicht gefunden")
        checks_passed = False
    
    # Prüfe Python-Module
    required_modules = ["duckdb", "google.generativeai", "openai", "anthropic"]
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"{Colors.WARNING}⚠{Colors.ENDC} Fehlende Python-Module: {', '.join(missing_modules)}")
        print(f"  Installiere mit: pip install {' '.join(missing_modules)}")
        checks_passed = False
    else:
        print(f"{Colors.OKGREEN}✓{Colors.ENDC} Alle Python-Module installiert")
    
    return checks_passed

def prepare_test_database():
    """Überprüft die Test-Datenbank"""
    print_section("📊 Datenbank-Vorbereitung")
    
    # Nutze direkt die Hauptdatenbank
    conn = duckdb.connect(INPUT_DB)
    
    try:
        # Zähle Speeches
        total_speeches = conn.execute("SELECT COUNT(*) FROM speeches").fetchone()[0]
        print(f"  Gesamt Speeches in DB: {total_speeches:,}")
        
        # Prüfe ob LLM-Spalten existieren und füge sie hinzu falls nötig
        try:
            conn.execute("SELECT llm_processed FROM speeches LIMIT 1")
            print(f"  ✅ LLM-Spalten bereits vorhanden")
        except:
            print(f"  🔧 Füge LLM-Spalten hinzu...")
            conn.execute("""
                ALTER TABLE speeches ADD COLUMN IF NOT EXISTS llm_processed BOOLEAN DEFAULT FALSE;
                ALTER TABLE speeches ADD COLUMN IF NOT EXISTS llm_classified_brexit BOOLEAN DEFAULT FALSE;
                ALTER TABLE speeches ADD COLUMN IF NOT EXISTS llm_confidence_score FLOAT;
                ALTER TABLE speeches ADD COLUMN IF NOT EXISTS llm_key_indicators VARCHAR;
                ALTER TABLE speeches ADD COLUMN IF NOT EXISTS llm_provider VARCHAR;
                ALTER TABLE speeches ADD COLUMN IF NOT EXISTS llm_model VARCHAR;
                ALTER TABLE speeches ADD COLUMN IF NOT EXISTS llm_cost_usd FLOAT;
                ALTER TABLE speeches ADD COLUMN IF NOT EXISTS llm_processing_time FLOAT;
                ALTER TABLE speeches ADD COLUMN IF NOT EXISTS llm_input_tokens INTEGER;
                ALTER TABLE speeches ADD COLUMN IF NOT EXISTS llm_output_tokens INTEGER;
                ALTER TABLE speeches ADD COLUMN IF NOT EXISTS llm_error TEXT;
                ALTER TABLE speeches ADD COLUMN IF NOT EXISTS llm_processed_at TIMESTAMP;
                ALTER TABLE speeches ADD COLUMN IF NOT EXISTS llm_reasoning TEXT;
            """)
            print(f"  ✅ LLM-Spalten hinzugefügt")
        
        # Prüfe ob bereits verarbeitete Speeches vorhanden sind
        processed_count = conn.execute("SELECT COUNT(*) FROM speeches WHERE llm_processed = TRUE").fetchone()[0]
        print(f"  Bereits verarbeitete Speeches: {processed_count:,}")
        
        # Zähle unverarbeitete Speeches für Test
        unprocessed_count = conn.execute("SELECT COUNT(*) FROM speeches WHERE llm_processed IS NULL OR llm_processed = FALSE").fetchone()[0]
        print(f"  Verfügbare Speeches für Test: {unprocessed_count:,}")
        
        if unprocessed_count < TEST_SIZE:
            print(f"  {Colors.WARNING}⚠ Warnung: Nur {unprocessed_count} unverarbeitete Speeches verfügbar, weniger als gewünschte {TEST_SIZE}{Colors.ENDC}")
        
        return total_speeches
        
    finally:
        conn.close()

def run_classification_test():
    """Führt den Klassifikations-Test aus"""
    print_section("🚀 Starte Klassifikations-Test")
    
    print(f"\n  Budget-Limit für Test: $2.00")
    print(f"  Speeches zu klassifizieren: {TEST_SIZE}")
    print(f"\n  {Colors.WARNING}Starte Hauptskript...{Colors.ENDC}")
    
    # Erstelle modifiziertes Test-Skript das direkt die Hauptdatenbank nutzt
    test_script = f"""
import sys
import os
sys.path.insert(0, '.')

# Setze Umgebungsvariablen für Test
os.environ['INPUT_DB'] = '{INPUT_DB}'
os.environ['OUTPUT_DB'] = '{TEST_DB}'
os.environ['MAX_BUDGET_USD'] = '2.00'
os.environ['BATCH_SIZE'] = '50'
os.environ['TEST_MODE'] = 'true'
os.environ['TEST_LIMIT'] = '{TEST_SIZE}'

# Importiere und führe Hauptskript aus
try:
    from classify_brexit_with_gemini import main
    main()
except Exception as e:
    print(f"Fehler beim Ausführen: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    
    with open("run_test.py", 'w') as f:
        f.write(test_script)
    
    start_time = time.time()
    
    try:
        # Führe Test aus
        result = subprocess.run(
            [sys.executable, "run_test.py"],
            capture_output=True,
            text=True,
            timeout=600  # 10 Minuten Timeout
        )
        
        duration = time.time() - start_time
        
        # Zeige Output
        if result.stdout:
            print(f"\n{Colors.OKBLUE}Output:{Colors.ENDC}")
            for line in result.stdout.split('\n')[-20:]:  # Letzte 20 Zeilen
                if line:
                    print(f"  {line}")
        
        if result.stderr:
            print(f"\n{Colors.WARNING}Warnungen:{Colors.ENDC}")
            for line in result.stderr.split('\n')[-10:]:  # Letzte 10 Zeilen
                if line:
                    print(f"  {line}")
        
        return duration, result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"{Colors.FAIL}✗ Test-Timeout nach 10 Minuten{Colors.ENDC}")
        return 600, False
    except Exception as e:
        print(f"{Colors.FAIL}✗ Fehler beim Test: {e}{Colors.ENDC}")
        return 0, False
    finally:
        # Aufräumen
        if Path("run_test.py").exists():
            Path("run_test.py").unlink()

def analyze_test_results():
    """Analysiert die Test-Ergebnisse"""
    print_section("📈 Test-Ergebnis-Analyse")
    
    if not Path(INPUT_DB).exists():
        print(f"{Colors.FAIL}✗ Datenbank nicht gefunden{Colors.ENDC}")
        return None
    
    conn = duckdb.connect(INPUT_DB, read_only=True)
    
    try:
        # Sammle Statistiken
        stats = {}
        
        # Grundlegende Zahlen
        stats['total_speeches'] = conn.execute("SELECT COUNT(*) FROM speeches").fetchone()[0]
        stats['processed'] = conn.execute("SELECT COUNT(*) FROM speeches WHERE llm_processed = TRUE").fetchone()[0]
        stats['brexit_classified'] = conn.execute("SELECT COUNT(*) FROM speeches WHERE llm_classified_brexit = TRUE").fetchone()[0]
        stats['errors'] = conn.execute("SELECT COUNT(*) FROM speeches WHERE llm_error IS NOT NULL").fetchone()[0]
        
        # Kosten-Analyse
        cost_data = conn.execute("""
            SELECT 
                SUM(llm_cost_usd) as total_cost,
                AVG(llm_cost_usd) as avg_cost,
                MIN(llm_cost_usd) as min_cost,
                MAX(llm_cost_usd) as max_cost
            FROM speeches 
            WHERE llm_processed = TRUE
        """).fetchone()
        
        stats['total_cost'] = cost_data[0] or 0
        stats['avg_cost_per_speech'] = cost_data[1] or 0
        stats['min_cost'] = cost_data[2] or 0
        stats['max_cost'] = cost_data[3] or 0
        
        # Provider-Verteilung
        provider_stats = conn.execute("""
            SELECT 
                llm_provider,
                COUNT(*) as count,
                SUM(llm_cost_usd) as cost,
                AVG(llm_processing_time) as avg_time
            FROM speeches 
            WHERE llm_processed = TRUE
            GROUP BY llm_provider
        """).fetchall()
        
        stats['provider_distribution'] = {
            row[0]: {
                'count': row[1],
                'cost': row[2] or 0,
                'avg_time': row[3] or 0
            } for row in provider_stats if row[0]
        }
        
        # Model-Verteilung
        model_stats = conn.execute("""
            SELECT 
                llm_model,
                COUNT(*) as count,
                SUM(llm_cost_usd) as cost
            FROM speeches 
            WHERE llm_processed = TRUE
            GROUP BY llm_model
        """).fetchall()
        
        stats['model_distribution'] = {
            row[0]: {
                'count': row[1],
                'cost': row[2] or 0
            } for row in model_stats if row[0]
        }
        
        # Zeit-Analyse
        time_data = conn.execute("""
            SELECT 
                AVG(llm_processing_time) as avg_time,
                MIN(llm_processing_time) as min_time,
                MAX(llm_processing_time) as max_time,
                SUM(llm_processing_time) as total_time
            FROM speeches 
            WHERE llm_processed = TRUE
        """).fetchone()
        
        stats['avg_processing_time'] = time_data[0] or 0
        stats['min_processing_time'] = time_data[1] or 0
        stats['max_processing_time'] = time_data[2] or 0
        stats['total_processing_time'] = time_data[3] or 0
        
        # Token-Statistiken
        token_data = conn.execute("""
            SELECT 
                AVG(llm_input_tokens) as avg_input,
                AVG(llm_output_tokens) as avg_output,
                SUM(llm_input_tokens) as total_input,
                SUM(llm_output_tokens) as total_output
            FROM speeches 
            WHERE llm_processed = TRUE
        """).fetchone()
        
        stats['avg_input_tokens'] = token_data[0] or 0
        stats['avg_output_tokens'] = token_data[1] or 0
        stats['total_input_tokens'] = token_data[2] or 0
        stats['total_output_tokens'] = token_data[3] or 0
        
        # Confidence Score Verteilung
        confidence_data = conn.execute("""
            SELECT 
                AVG(llm_confidence_score) as avg_conf,
                MIN(llm_confidence_score) as min_conf,
                MAX(llm_confidence_score) as max_conf
            FROM speeches 
            WHERE llm_processed = TRUE
        """).fetchone()
        
        stats['avg_confidence'] = confidence_data[0] or 0
        stats['min_confidence'] = confidence_data[1] or 0
        stats['max_confidence'] = confidence_data[2] or 0
        
        return stats
        
    finally:
        conn.close()

def print_test_results(stats, test_duration, total_speeches_in_db):
    """Zeigt die Test-Ergebnisse formatiert an"""
    print_header("TEST-ERGEBNISSE")
    
    # Basis-Statistiken
    print_section("📊 Basis-Statistiken")
    print(f"  Verarbeitete Speeches:  {stats['processed']:,}")
    print(f"  Brexit-Speeches:        {stats['brexit_classified']:,} ({stats['brexit_classified']/max(stats['processed'],1)*100:.1f}%)")
    print(f"  Fehler:                 {stats['errors']:,}")
    print(f"  Test-Dauer:             {format_duration(test_duration)}")
    
    # Kosten-Analyse
    print_section("💰 Kosten-Analyse")
    print(f"  Gesamtkosten Test:      {format_currency(stats['total_cost'])}")
    print(f"  Kosten pro Speech:      {format_currency(stats['avg_cost_per_speech'])}")
    print(f"  Min/Max Kosten:         {format_currency(stats['min_cost'])} - {format_currency(stats['max_cost'])}")
    print(f"  Kosten pro 1k Speeches: {format_currency(stats['avg_cost_per_speech'] * 1000)}")
    
    # Performance-Analyse
    print_section("⚡ Performance-Analyse")
    if test_duration > 0:
        speeches_per_second = stats['processed'] / test_duration
        print(f"  Speeches/Sekunde:       {speeches_per_second:.2f}")
        print(f"  Speeches/Minute:        {speeches_per_second * 60:.0f}")
        print(f"  Speeches/Stunde:        {speeches_per_second * 3600:.0f}")
    
    print(f"  Ø Verarbeitungszeit:    {stats['avg_processing_time']:.2f}s")
    print(f"  Min/Max Zeit:           {stats['min_processing_time']:.2f}s - {stats['max_processing_time']:.2f}s")
    
    # Token-Statistiken
    print_section("📝 Token-Statistiken")
    print(f"  Ø Input Tokens:         {stats['avg_input_tokens']:.0f}")
    print(f"  Ø Output Tokens:        {stats['avg_output_tokens']:.0f}")
    print(f"  Total Tokens:           {(stats['total_input_tokens'] + stats['total_output_tokens']):,.0f}")
    
    # Provider-Verteilung
    if stats['provider_distribution']:
        print_section("🔧 Provider-Verteilung")
        for provider, data in sorted(stats['provider_distribution'].items(), 
                                    key=lambda x: x[1]['count'], reverse=True):
            percentage = data['count'] / stats['processed'] * 100
            print(f"  {provider:15} {data['count']:4} speeches ({percentage:5.1f}%) | "
                  f"Kosten: {format_currency(data['cost'])} | "
                  f"Ø Zeit: {data['avg_time']:.2f}s")
    
    # Model-Verteilung (Top 5)
    if stats['model_distribution']:
        print_section("🤖 Top 5 Modelle")
        sorted_models = sorted(stats['model_distribution'].items(), 
                              key=lambda x: x[1]['count'], reverse=True)[:5]
        for model, data in sorted_models:
            model_name = model.split('/')[-1] if '/' in model else model
            percentage = data['count'] / stats['processed'] * 100
            avg_cost = data['cost'] / data['count'] if data['count'] > 0 else 0
            print(f"  {model_name[:30]:30} {data['count']:4} ({percentage:5.1f}%) | "
                  f"Ø: {format_currency(avg_cost)}")

def calculate_projections(stats, test_duration, total_speeches_in_db):
    """Berechnet Hochrechnungen für alle Speeches"""
    print_header("HOCHRECHNUNG FÜR ALLE SPEECHES")
    
    if stats['processed'] == 0:
        print(f"{Colors.FAIL}Keine Daten für Hochrechnung verfügbar{Colors.ENDC}")
        return
    
    # Berechne Faktoren
    if stats['processed'] > 0:
        scale_factor = TOTAL_SPEECHES_TARGET / stats['processed']
    else:
        print(f"{Colors.FAIL}Keine verarbeiteten Speeches für Hochrechnung{Colors.ENDC}")
        return
    
    # Kosten-Hochrechnung
    print_section(f"💰 Kosten-Hochrechnung für {TOTAL_SPEECHES_TARGET:,} Speeches")
    
    projected_cost = stats['avg_cost_per_speech'] * TOTAL_SPEECHES_TARGET
    cost_range_min = stats['min_cost'] * TOTAL_SPEECHES_TARGET
    cost_range_max = stats['max_cost'] * TOTAL_SPEECHES_TARGET
    
    print(f"  Geschätzte Gesamtkosten:    {format_currency(projected_cost)}")
    print(f"  Kostenbereich:               {format_currency(cost_range_min)} - {format_currency(cost_range_max)}")
    print(f"  Budget $20 reicht für:       {int(20 / stats['avg_cost_per_speech']):,} Speeches")
    
    # Zeit-Hochrechnung
    print_section(f"⏱️ Zeit-Hochrechnung für {TOTAL_SPEECHES_TARGET:,} Speeches")
    
    if test_duration > 0:
        speeches_per_second = stats['processed'] / test_duration
        projected_duration = TOTAL_SPEECHES_TARGET / speeches_per_second
        
        print(f"  Geschätzte Gesamtdauer:     {format_duration(projected_duration)}")
        
        # Verschiedene Szenarien
        print(f"\n  Szenarien mit Pausen/Delays:")
        print(f"    Optimistisch (100%):     {format_duration(projected_duration)}")
        print(f"    Realistisch (150%):      {format_duration(projected_duration * 1.5)}")
        print(f"    Pessimistisch (200%):    {format_duration(projected_duration * 2)}")
    
    # Token-Hochrechnung
    print_section("📊 Token-Hochrechnung")
    
    total_tokens = (stats['avg_input_tokens'] + stats['avg_output_tokens']) * TOTAL_SPEECHES_TARGET
    print(f"  Geschätzte Total Tokens:    {total_tokens/1_000_000:.1f}M")
    print(f"  Input Tokens:                {stats['avg_input_tokens'] * TOTAL_SPEECHES_TARGET / 1_000_000:.1f}M")
    print(f"  Output Tokens:               {stats['avg_output_tokens'] * TOTAL_SPEECHES_TARGET / 1_000_000:.1f}M")
    
    # Provider-Empfehlung
    print_section("💡 Empfehlungen")
    
    # Finde günstigsten Provider
    if stats['provider_distribution']:
        cheapest_provider = min(stats['provider_distribution'].items(),
                               key=lambda x: x[1]['cost']/x[1]['count'] if x[1]['count'] > 0 else float('inf'))
        fastest_provider = min(stats['provider_distribution'].items(),
                              key=lambda x: x[1]['avg_time'])
        
        print(f"  Günstigster Provider:        {cheapest_provider[0]} "
              f"(Ø {format_currency(cheapest_provider[1]['cost']/cheapest_provider[1]['count'])} pro Speech)")
        print(f"  Schnellster Provider:        {fastest_provider[0]} "
              f"(Ø {fastest_provider[1]['avg_time']:.2f}s pro Speech)")
    
    # Budget-Empfehlungen
    print(f"\n  Budget-Empfehlungen:")
    print(f"    Minimum (nur günstigste):  {format_currency(projected_cost * 0.8)}")
    print(f"    Empfohlen (mit Puffer):    {format_currency(projected_cost * 1.2)}")
    print(f"    Sicher (mit Reserve):      {format_currency(projected_cost * 1.5)}")
    
    # Weitere Optimierungen
    print(f"\n  Optimierungsvorschläge:")
    if stats['avg_input_tokens'] > 500:
        print(f"    • Prompts kürzen (aktuell Ø {stats['avg_input_tokens']:.0f} Tokens)")
    if stats['errors'] > stats['processed'] * 0.05:
        print(f"    • Fehlerrate reduzieren (aktuell {stats['errors']/stats['processed']*100:.1f}%)")
    if len(stats['provider_distribution']) < 3:
        print(f"    • Mehr Provider aktivieren für bessere Lastverteilung")

def save_report(stats, test_duration, total_speeches_in_db):
    """Speichert einen detaillierten Report"""
    report_path = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    report = {
        'test_metadata': {
            'timestamp': datetime.now().isoformat(),
            'test_size': TEST_SIZE,
            'test_duration_seconds': test_duration,
            'total_speeches_in_db': total_speeches_in_db
        },
        'results': stats,
        'projections': {
            'target_speeches': TOTAL_SPEECHES_TARGET,
            'estimated_cost': stats['avg_cost_per_speech'] * TOTAL_SPEECHES_TARGET,
            'estimated_duration_seconds': (TOTAL_SPEECHES_TARGET / stats['processed']) * test_duration if stats['processed'] > 0 else 0,
            'speeches_per_dollar': int(1 / stats['avg_cost_per_speech']) if stats['avg_cost_per_speech'] > 0 else 0
        }
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n{Colors.OKGREEN}✓ Detaillierter Report gespeichert: {report_path}{Colors.ENDC}")

def main():
    """Hauptfunktion für Test"""
    print_header("LLM KLASSIFIKATIONS-TEST")
    print(f"{Colors.BOLD}Testet mit {TEST_SIZE} Speeches und rechnet auf {TOTAL_SPEECHES_TARGET:,} hoch{Colors.ENDC}")
    
    # Schritt 1: Umgebungsprüfung
    if not check_environment():
        print(f"\n{Colors.FAIL}✗ Umgebungsprüfung fehlgeschlagen. Bitte Probleme beheben.{Colors.ENDC}")
        return 1
    
    # Schritt 2: Datenbank vorbereiten
    total_speeches_in_db = prepare_test_database()
    
    # Schritt 3: Test ausführen
    test_duration, success = run_classification_test()
    
    if not success:
        print(f"\n{Colors.FAIL}✗ Test fehlgeschlagen{Colors.ENDC}")
        return 1
    
    # Schritt 4: Ergebnisse analysieren
    stats = analyze_test_results()
    
    if not stats:
        print(f"\n{Colors.FAIL}✗ Keine Ergebnisse zum Analysieren{Colors.ENDC}")
        return 1
    
    # Schritt 5: Ergebnisse anzeigen
    print_test_results(stats, test_duration, total_speeches_in_db)
    
    # Schritt 6: Hochrechnung
    calculate_projections(stats, test_duration, total_speeches_in_db)
    
    # Schritt 7: Report speichern
    save_report(stats, test_duration, total_speeches_in_db)
    
    # Zusammenfassung
    print_header("ZUSAMMENFASSUNG")
    print(f"{Colors.OKGREEN}✓ Test erfolgreich abgeschlossen!{Colors.ENDC}")
    print(f"\n  Getestet:    {stats['processed']:,} Speeches")
    print(f"  Kosten:      {format_currency(stats['total_cost'])}")
    print(f"  Dauer:       {format_duration(test_duration)}")
    print(f"\n  Hochrechnung für {TOTAL_SPEECHES_TARGET:,} Speeches:")
    print(f"  → Kosten:    {format_currency(stats['avg_cost_per_speech'] * TOTAL_SPEECHES_TARGET)}")
    if stats['processed'] > 0:
        print(f"  → Dauer:     {format_duration((TOTAL_SPEECHES_TARGET / stats['processed']) * test_duration)}")
    else:
        print(f"  → Dauer:     N/A (keine Speeches verarbeitet)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())