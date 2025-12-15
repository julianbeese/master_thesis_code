"""
Brexit Speech Classifier mit vollständiger LLM-Metadaten-Erfassung
Klassifiziert ~40.000 Speeches und füllt alle LLM-Spalten aus
"""

import asyncio
import duckdb
from openai import AsyncOpenAI
import os
from datetime import datetime
from typing import List, Dict, Optional
import json
from dataclasses import dataclass
import logging
from collections import defaultdict
import time
import re

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('brexit_classification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Konfiguration für ein OpenAI-Modell"""
    name: str
    rpm_limit: int
    tpm_limit: int
    cost_per_1k_input: float
    cost_per_1k_output: float


# Günstige OpenAI-Modelle
MODELS = [
    ModelConfig("gpt-4o-mini", 500, 200000, 0.00015, 0.0006),
    ModelConfig("gpt-3.5-turbo", 500, 160000, 0.0005, 0.0015),
]


class RateLimiter:
    """Token-Bucket Rate Limiter für API-Calls"""
    
    def __init__(self, rpm_limit: int, tpm_limit: int):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.request_times = []
        self.token_count = 0
        self.token_reset_time = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self, estimated_tokens: int = 500):
        """Warte bis Request möglich ist"""
        async with self.lock:
            now = time.time()
            
            if now - self.token_reset_time >= 60:
                self.token_count = 0
                self.token_reset_time = now
            
            self.request_times = [t for t in self.request_times if now - t < 60]
            
            while (len(self.request_times) >= self.rpm_limit or 
                   self.token_count + estimated_tokens > self.tpm_limit):
                await asyncio.sleep(0.5)
                now = time.time()
                
                if now - self.token_reset_time >= 60:
                    self.token_count = 0
                    self.token_reset_time = now
                
                self.request_times = [t for t in self.request_times if now - t < 60]
            
            self.request_times.append(now)
            self.token_count += estimated_tokens


class BrexitClassifier:
    """Hauptklasse für die Brexit-Klassifizierung"""
    
    def __init__(self, db_path: str, api_key: str, checkpoint_file: str = "checkpoint.json"):
        self.db_path = db_path
        self.checkpoint_file = checkpoint_file
        self.client = AsyncOpenAI(api_key=api_key)
        
        self.rate_limiters = {
            model.name: RateLimiter(model.rpm_limit, model.tpm_limit)
            for model in MODELS
        }
        
        self.stats = defaultdict(int)
        self.start_time = None
        self.processed_ids = self.load_checkpoint()
        
        # Modell-Kosten für Berechnung
        self.model_costs = {m.name: m for m in MODELS}
    
    def load_checkpoint(self) -> set:
        """Lade bereits verarbeitete IDs"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                logger.info(f"Checkpoint geladen: {len(data['processed_ids'])} bereits verarbeitet")
                return set(data['processed_ids'])
        return set()
    
    def save_checkpoint(self):
        """Speichere Fortschritt"""
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                'processed_ids': list(self.processed_ids),
                'timestamp': datetime.now().isoformat(),
                'stats': dict(self.stats)
            }, f)
    
    def ensure_llm_columns(self):
        """Stelle sicher, dass alle LLM-Spalten existieren"""
        conn = duckdb.connect(self.db_path)
        
        columns_to_add = [
            ("llm_processed", "BOOLEAN"),
            ("llm_classified_brexit", "BOOLEAN"),
            ("llm_confidence_score", "DOUBLE"),
            ("llm_key_indicators", "VARCHAR"),
            ("llm_provider", "VARCHAR"),
            ("llm_model", "VARCHAR"),
            ("llm_cost_usd", "DOUBLE"),
            ("llm_processing_time", "DOUBLE"),
            ("llm_input_tokens", "INTEGER"),
            ("llm_output_tokens", "INTEGER"),
            ("llm_error", "VARCHAR"),
            ("llm_processed_at", "TIMESTAMP"),
            ("llm_reasoning", "VARCHAR"),
            ("llm_retry_count", "INTEGER"),
        ]
        
        for col_name, col_type in columns_to_add:
            try:
                conn.execute(f"SELECT {col_name} FROM speeches LIMIT 1")
            except:
                logger.info(f"Erstelle Spalte: {col_name}")
                conn.execute(f"ALTER TABLE speeches ADD COLUMN {col_name} {col_type}")
        
        conn.close()
    
    def get_speeches_to_classify(self) -> List[Dict]:
        """Hole alle noch nicht klassifizierten Speeches"""
        conn = duckdb.connect(self.db_path)
        
        # Hole Speeches die noch nicht verarbeitet wurden
        # Hauptkriterium: llm_processed muss NULL oder FALSE sein
        query = """
            SELECT speech_id, speech_text, speaker_name, time, speaker_office
            FROM speeches
            WHERE llm_processed IS NULL OR llm_processed = FALSE
        """
        
        speeches = conn.execute(query).fetchall()
        conn.close()
        
        # Filtere bereits im Checkpoint verarbeitete
        speeches = [s for s in speeches if s[0] not in self.processed_ids]
        
        logger.info(f"Gefunden: {len(speeches)} Speeches zu klassifizieren")
        return [
            {
                'id': s[0],
                'speech': s[1],
                'speaker_name': s[2],
                'time': s[3],
                'office': s[4]
            }
            for s in speeches
        ]
    
    async def classify_single_speech(
        self, 
        speech_data: Dict, 
        model_name: str,
        semaphore: asyncio.Semaphore,
        retry_count: int = 0
    ) -> Optional[Dict]:
        """Klassifiziere eine einzelne Speech mit vollständigen Metadaten"""
        async with semaphore:
            # Zusätzliche Sicherheitsprüfung: Prüfe ob bereits verarbeitet
            if speech_data['id'] in self.processed_ids:
                logger.warning(f"Speech {speech_data['id']} bereits im Checkpoint - überspringe")
                return None
                
            # Prüfe auch in der Datenbank
            conn = duckdb.connect(self.db_path)
            check_query = """
                SELECT llm_processed, llm_processed_at 
                FROM speeches 
                WHERE speech_id = ? AND (llm_processed = TRUE OR llm_processed_at IS NOT NULL)
            """
            result = conn.execute(check_query, [speech_data['id']]).fetchone()
            conn.close()
            
            if result:
                logger.warning(f"Speech {speech_data['id']} bereits in DB verarbeitet - überspringe")
                return None
            
            start_time = time.time()
            
            try:
                # Schätze Token-Anzahl
                estimated_tokens = len(speech_data['speech'].split()) * 1.3 + 300
                
                await self.rate_limiters[model_name].acquire(int(estimated_tokens))
                
                # Erstelle erweiterten Prompt
                prompt = self._create_detailed_prompt(speech_data)
                
                # API Call
                response = await self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a political analyst specialized in Brexit-related discourse. Analyze speeches carefully and provide detailed reasoning."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=150
                )
                
                processing_time = time.time() - start_time
                
                # Parse Response
                answer_text = response.choices[0].message.content.strip()
                classification_result = self._parse_llm_response(answer_text)
                
                # Berechne Kosten
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                model_config = self.model_costs[model_name]
                cost = (input_tokens * model_config.cost_per_1k_input / 1000 + 
                       output_tokens * model_config.cost_per_1k_output / 1000)
                
                self.stats['successful'] += 1
                self.stats[f'model_{model_name}'] += 1
                self.stats['total_cost'] += cost
                
                return {
                    'id': speech_data['id'],
                    'llm_processed': True,
                    'llm_classified_brexit': classification_result['is_brexit'],
                    'llm_confidence_score': classification_result['confidence'],
                    'llm_key_indicators': classification_result['indicators'],
                    'llm_provider': 'OpenAI',
                    'llm_model': model_name,
                    'llm_cost_usd': cost,
                    'llm_processing_time': processing_time,
                    'llm_input_tokens': input_tokens,
                    'llm_output_tokens': output_tokens,
                    'llm_error': None,
                    'llm_processed_at': datetime.now(),
                    'llm_reasoning': classification_result['reasoning'],
                    'llm_retry_count': retry_count
                }
                
            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(f"Fehler bei Speech {speech_data['id']}: {e}")
                self.stats['errors'] += 1
                
                # Retry Logic (max 2 Versuche)
                if retry_count < 2:
                    await asyncio.sleep(1)
                    return await self.classify_single_speech(
                        speech_data, model_name, semaphore, retry_count + 1
                    )
                
                # Fehlerfall speichern
                return {
                    'id': speech_data['id'],
                    'llm_processed': True,
                    'llm_classified_brexit': None,
                    'llm_confidence_score': None,
                    'llm_key_indicators': None,
                    'llm_provider': 'OpenAI',
                    'llm_model': model_name,
                    'llm_cost_usd': 0.0,
                    'llm_processing_time': processing_time,
                    'llm_input_tokens': None,
                    'llm_output_tokens': None,
                    'llm_error': str(e)[:500],
                    'llm_processed_at': datetime.now(),
                    'llm_reasoning': None,
                    'llm_retry_count': retry_count
                }
    
    def _create_detailed_prompt(self, speech_data: Dict) -> str:
        """Erstelle detaillierten Klassifizierungs-Prompt"""
        speech_excerpt = speech_data['speech'][:2000]
        
        return f"""Analyze this UK parliamentary speech for Brexit relevance.

SPEECH EXCERPT:
"{speech_excerpt}"

METADATA:
- Speaker: {speech_data['speaker_name']}
- Office: {speech_data['office']}
- Time: {speech_data['time']}

TASK:
Determine if this speech substantially discusses Brexit or UK-EU relations.

Brexit-relevant topics include:
- EU membership, referendum, withdrawal
- Article 50, withdrawal agreements
- Brexit consequences (trade, immigration, economy, regulations)
- UK-EU relationship in Brexit context
- No-deal Brexit scenarios

Provide your analysis in this EXACT format:
CLASSIFICATION: [YES or NO]
CONFIDENCE: [0.0-1.0]
KEY_INDICATORS: [comma-separated list of Brexit-related terms found, or "none"]
REASONING: [one sentence explanation]

Example:
CLASSIFICATION: YES
CONFIDENCE: 0.95
KEY_INDICATORS: withdrawal agreement, Article 50, leaving the EU
REASONING: Speech directly debates the withdrawal agreement and Article 50 process."""
    
    def _parse_llm_response(self, response_text: str) -> Dict:
        """Parse LLM Response in strukturiertes Format"""
        try:
            # Parse Classification
            is_brexit = 'YES' in response_text.upper().split('CLASSIFICATION:')[1].split('\n')[0]
            
            # Parse Confidence
            confidence_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response_text, re.IGNORECASE)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5
            confidence = max(0.0, min(1.0, confidence))
            
            # Parse Key Indicators
            indicators_match = re.search(r'KEY_INDICATORS:\s*(.+?)(?:\n|$)', response_text, re.IGNORECASE)
            indicators = indicators_match.group(1).strip() if indicators_match else "none"
            if indicators.lower() in ['none', 'n/a', '']:
                indicators = None
            
            # Parse Reasoning
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?:\n|$)', response_text, re.IGNORECASE | re.DOTALL)
            reasoning = reasoning_match.group(1).strip()[:500] if reasoning_match else None
            
            return {
                'is_brexit': is_brexit,
                'confidence': confidence,
                'indicators': indicators,
                'reasoning': reasoning
            }
        except Exception as e:
            logger.warning(f"Fehler beim Parsen der LLM-Antwort: {e}")
            # Fallback
            is_brexit = 'YES' in response_text.upper()
            return {
                'is_brexit': is_brexit,
                'confidence': 0.5,
                'indicators': None,
                'reasoning': response_text[:500]
            }
    
    async def classify_batch(
        self, 
        speeches: List[Dict], 
        batch_size: int = 100,
        concurrent_requests: int = 50
    ):
        """Klassifiziere Batch von Speeches"""
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        for i in range(0, len(speeches), batch_size):
            batch = speeches[i:i + batch_size]
            
            # Round-Robin auf Modelle verteilen
            tasks = []
            for idx, speech in enumerate(batch):
                model = MODELS[idx % len(MODELS)]
                task = self.classify_single_speech(speech, model.name, semaphore)
                tasks.append(task)
            
            # Parallel ausführen
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Ergebnisse speichern
            valid_results = [r for r in results if r and not isinstance(r, Exception)]
            if valid_results:
                try:
                    # Speichere zuerst in DB
                    self.save_results_to_db(valid_results)
                    # Dann erst Checkpoint aktualisieren
                    self.processed_ids.update([r['id'] for r in valid_results])
                    logger.info(f"Batch {i//batch_size + 1}: {len(valid_results)} Speeches erfolgreich verarbeitet")
                except Exception as e:
                    logger.error(f"Fehler beim Speichern von Batch {i//batch_size + 1}: {e}")
                    # Bei DB-Fehler: Checkpoint nicht aktualisieren
                    continue
            
            # Checkpoint alle 500 Speeches
            if i % 500 == 0:
                self.save_checkpoint()
                self.log_progress(i + len(batch), len(speeches))
    
    def save_results_to_db(self, results: List[Dict]):
        """Speichere vollständige Klassifizierungen in DB mit atomarer Sicherheit"""
        conn = duckdb.connect(self.db_path)
        
        try:
            # Starte Transaktion für atomare Updates
            conn.execute("BEGIN TRANSACTION")
            
            for result in results:
                # Zusätzliche Sicherheitsprüfung vor Update
                check_query = """
                    SELECT llm_processed, llm_processed_at 
                    FROM speeches 
                    WHERE speech_id = ? AND (llm_processed = TRUE OR llm_processed_at IS NOT NULL)
                """
                existing = conn.execute(check_query, [result['id']]).fetchone()
                
                if existing:
                    logger.warning(f"Speech {result['id']} bereits verarbeitet - überspringe DB-Update")
                    continue
                
                # Führe Update nur aus, wenn noch nicht verarbeitet
                conn.execute("""
                    UPDATE speeches 
                    SET 
                        llm_processed = ?,
                        llm_classified_brexit = ?,
                        llm_confidence_score = ?,
                        llm_key_indicators = ?,
                        llm_provider = ?,
                        llm_model = ?,
                        llm_cost_usd = ?,
                        llm_processing_time = ?,
                        llm_input_tokens = ?,
                        llm_output_tokens = ?,
                        llm_error = ?,
                        llm_processed_at = ?,
                        llm_reasoning = ?,
                        llm_retry_count = ?
                    WHERE speech_id = ? AND (llm_processed IS NULL OR llm_processed = FALSE)
                """, [
                    result['llm_processed'],
                    result['llm_classified_brexit'],
                    result['llm_confidence_score'],
                    result['llm_key_indicators'],
                    result['llm_provider'],
                    result['llm_model'],
                    result['llm_cost_usd'],
                    result['llm_processing_time'],
                    result['llm_input_tokens'],
                    result['llm_output_tokens'],
                    result['llm_error'],
                    result['llm_processed_at'],
                    result['llm_reasoning'],
                    result['llm_retry_count'],
                    result['id']
                ])
            
            # Commit Transaktion
            conn.execute("COMMIT")
            logger.info(f"Erfolgreich {len(results)} Ergebnisse in DB gespeichert")
            
        except Exception as e:
            # Rollback bei Fehler
            conn.execute("ROLLBACK")
            logger.error(f"Fehler beim Speichern in DB: {e}")
            raise
        finally:
            conn.close()
    
    def verify_no_duplicates(self):
        """Überprüfe auf doppelt verarbeitete Speeches"""
        conn = duckdb.connect(self.db_path)
        
        # Prüfe auf Duplikate in der DB
        duplicate_query = """
            SELECT speech_id, COUNT(*) as count
            FROM speeches 
            WHERE llm_processed = TRUE
            GROUP BY speech_id 
            HAVING COUNT(*) > 1
        """
        duplicates = conn.execute(duplicate_query).fetchall()
        
        if duplicates:
            logger.error(f"GEFUNDEN: {len(duplicates)} doppelt verarbeitete Speeches!")
            for speech_id, count in duplicates:
                logger.error(f"Speech {speech_id}: {count} mal verarbeitet")
        else:
            logger.info("✓ Keine Duplikate in der Datenbank gefunden")
        
        conn.close()
        return len(duplicates) == 0

    def log_progress(self, current: int, total: int):
        """Logge detaillierten Fortschritt"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            rate = current / elapsed if elapsed > 0 else 0
            remaining = (total - current) / rate if rate > 0 else 0
            
            logger.info(f"""
Progress: {current}/{total} ({current/total*100:.1f}%)
Rate: {rate:.1f} speeches/sec
Elapsed: {elapsed/3600:.2f}h
Estimated remaining: {remaining/3600:.2f}h
Success: {self.stats['successful']} | Errors: {self.stats['errors']}
Total Cost: ${self.stats.get('total_cost', 0):.4f}
            """.strip())
    
    async def run(self):
        """Hauptausführung"""
        logger.info("Starte Brexit-Klassifizierung mit vollständiger LLM-Metadaten-Erfassung...")
        self.start_time = time.time()
        
        # Stelle Spalten sicher
        self.ensure_llm_columns()
        
        # Hole Speeches
        speeches = self.get_speeches_to_classify()
        
        if not speeches:
            logger.info("Keine Speeches zu klassifizieren!")
            return
        
        # Klassifiziere
        await self.classify_batch(speeches, concurrent_requests=50)
        
        # Finale Statistiken
        elapsed = time.time() - self.start_time
        logger.info(f"""
=== ABGESCHLOSSEN ===
Total verarbeitet: {self.stats['successful']}
Fehler: {self.stats['errors']}
Gesamtzeit: {elapsed/3600:.2f}h
Durchschnittsrate: {self.stats['successful']/elapsed:.1f} speeches/sec
Gesamtkosten: ${self.stats.get('total_cost', 0):.4f}
        """.strip())
        
        # Überprüfe auf Duplikate
        logger.info("Überprüfe auf doppelt verarbeitete Speeches...")
        self.verify_no_duplicates()
        
        self.save_checkpoint()


async def main():
    """Hauptfunktion"""
    # Lese API Key direkt aus .env Datei
    env_path = '/Users/julianbeese/Developer/Master/data_collection/.env'
    api_key = None
    
    with open(env_path, 'r') as f:
        for line in f:
            if line.startswith('OPENAI_API_KEY='):
                api_key = line.split('=', 1)[1].strip()
                break
    
    DB_PATH = "/Users/julianbeese/Developer/Master/data_collection/data/processed/brexit_analysis.duckdb"
    
    if not api_key or not api_key.startswith('sk-'):
        raise ValueError(f"OPENAI_API_KEY ungültig!")
    
    logger.info("API Key erfolgreich geladen!")
    logger.info("Starte vollständige LLM-Klassifizierung mit Metadaten...")
    
    # Starte Klassifizierung
    classifier = BrexitClassifier(DB_PATH, api_key)
    await classifier.run()


if __name__ == "__main__":
    asyncio.run(main())