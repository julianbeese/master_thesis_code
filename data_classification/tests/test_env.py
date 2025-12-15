from dotenv import load_dotenv
import os

print("Lade .env Datei...")
load_dotenv('/Users/julianbeese/Developer/Master/data_collection/.env')

print("\nAlle Umgebungsvariablen:")
for key in ['OPENAI_API_KEY', 'GEMINI_API_KEY', 'ANTHROPIC_API_KEY']:
    value = os.getenv(key)
    if value:
        print(f"{key}: {value[:30]}... (Länge: {len(value)})")
    else:
        print(f"{key}: NICHT GEFUNDEN")

# Teste auch ob die Datei existiert
import os.path
env_path = '/Users/julianbeese/Developer/Master/data_collection/.env'
print(f"\n.env Datei existiert: {os.path.exists(env_path)}")

# Zeige erste Zeilen der .env Datei
print("\nErste Zeilen der .env:")
with open(env_path, 'r') as f:
    for i, line in enumerate(f):
        if i < 10:
            # Maskiere sensitive Daten
            if '=' in line:
                key, val = line.split('=', 1)
                print(f"{key}={val[:20]}...")
            else:
                print(line.strip())