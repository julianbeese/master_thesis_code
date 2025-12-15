#!/usr/bin/env python3
"""
Klassifiziert Brexit-Speeches mit mehreren günstigen LLM-Providern
Mit Kostentracking, Budget-Limit und robuster Fehlerbehandlung
"""

import duckdb
import os
import time
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from queue import Queue, PriorityQueue
import threading
from datetime import datetime, timedelta
import random
from decimal import Decimal
import logging
from enum import Enum
import signal
from contextlib import contextmanager

# Optional imports für Provider
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None

# Debug: Zeige verfügbare Module
print(f"🔍 Provider-Verfügbarkeit:")
print(f"  Gemini: {'✅' if GEMINI_AVAILABLE else '❌'}")
print(f"  OpenAI: {'✅' if OPENAI_AVAILABLE else '❌'}")
print(f"  Anthropic: {'✅' if ANTHROPIC_AVAILABLE else '❌'}")

# Simple timeout wrapper using concurrent.futures
def with_timeout(func, timeout_seconds):
    """Execute function with timeout using ThreadPoolExecutor"""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            raise TimeoutError(f"Request timeout after {timeout_seconds}s")

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_classification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Konfiguration - kann durch Umgebungsvariablen überschrieben werden
INPUT_DB = os.getenv('INPUT_DB', "/Users/julianbeese/Developer/Master/data_collection/data/processed/brexit_analysis.duckdb")
OUTPUT_DB = os.getenv('OUTPUT_DB', "/Users/julianbeese/Developer/Master/data_collection/data/processed/brexit_multi_llm_classified.duckdb")

# Budget-Konfiguration
MAX_BUDGET_USD = float(os.getenv('MAX_BUDGET_USD', '20.00'))  # Maximum Budget in USD
COST_WARNING_THRESHOLD = 0.8  # Warnung bei 80% des Budgets

# Günstige Provider Konfiguration mit mehreren Modellen
PROVIDERS_CONFIG = {
    "gemini": {
        "models": [
            {
                "name": "gemini-2.5-flash-lite",
                "cost_per_1M_input": 0.075,
                "cost_per_1M_output": 0.30,
                "rpm_limit": 10,
                "tpm_limit": 32000,
                "rpd_limit": 1500,
                "request_timeout": 30
            },
            {
                "name": "gemini-2.5-flash",
                "cost_per_1M_input": 0.0375,
                "cost_per_1M_output": 0.15,
                "rpm_limit": 10,
                "tpm_limit": 32000,
                "rpd_limit": 1500,
                "request_timeout": 30
            },
        ],
        "enabled": True
    },
    "openai": {
        "models": [
            {
                "name": "gpt-4o-mini",
                "cost_per_1M_input": 0.15,
                "cost_per_1M_output": 0.60,
                "rpm_limit": 500,  # Tier 1
                "tpm_limit": 200000
            },
            {
                "name": "gpt-3.5-turbo",
                "cost_per_1M_input": 0.50,
                "cost_per_1M_output": 1.50,
                "rpm_limit": 500,
                "tpm_limit": 200000
            }
            # gpt-5-nano und gpt-5-mini deaktiviert - erzeugen leere Responses
            # {
            #     "name": "gpt-5-nano",
            #     "cost_per_1M_input": 0.15,
            #     "cost_per_1M_output": 0.60,
            #     "rpm_limit": 500,
            #     "tpm_limit": 200000
            # },
            # {
            #     "name": "gpt-5-mini",
            #     "cost_per_1M_input": 0.15,
            #     "cost_per_1M_output": 0.60,
            #     "rpm_limit": 500,
            #     "tpm_limit": 200000
            # }

        ],
        "enabled": True
    },
    "claude": {
        "models": [
            {
                "name": "claude-3-haiku-20240307",
                "cost_per_1M_input": 0.25,
                "cost_per_1M_output": 1.25,
                "rpm_limit": 50,  # Tier 1
                "tpm_limit": 50000,
                "tpd_limit": 5000000
            }
        ],
        "enabled": True
    },
    "deepseek": {
        "models": [
            {
                "name": "deepseek-chat",
                "cost_per_1M_input": 0.14,  # Off-peak pricing
                "cost_per_1M_output": 0.28,
                "rpm_limit": 100,
                "tpm_limit": 1000000
            }
        ],
        "enabled": True,
        "base_url": "https://api.deepseek.com/v1"
    },
    "mistral": {
        "models": [
            {
                "name": "open-mistral-7b",
                "cost_per_1M_input": 0.25,
                "cost_per_1M_output": 0.25,
                "rpm_limit": 100,
                "tpm_limit": 500000
            }
        ],
        "enabled": True
    },
    "together": {
        "models": [
            {
                "name": "mistralai/Mistral-7B-Instruct-v0.2",
                "cost_per_1M_input": 0.20,
                "cost_per_1M_output": 0.20,
                "rpm_limit": 600,
                "tpm_limit": 1000000
            },
            {
                "name": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
                "cost_per_1M_input": 0.06,
                "cost_per_1M_output": 0.06,
                "rpm_limit": 600,
                "tpm_limit": 1000000
            }
        ],
        "enabled": True,
        "base_url": "https://api.together.xyz/v1"
    }
}

# Batch Processing - Optimiert für bessere Performance
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '50'))  # Kleinere Batches für bessere Kontrolle
MAX_RETRIES = 3  # Weniger Retries für schnellere Fehlerbehandlung
INITIAL_RETRY_DELAY = 0.5  # Schnellere Retries
MAX_RETRY_DELAY = 30.0  # Kürzere Max-Delays
BACKOFF_FACTOR = 1.5  # Sanftere Backoffs

# Provider-spezifische Worker-Konfiguration
PROVIDER_WORKER_SAFETY_MARGIN = float(os.getenv('PROVIDER_WORKER_SAFETY_MARGIN', '0.8'))  # 80% der RPM-Limits nutzen
DYNAMIC_ADJUSTMENT_ENABLED = os.getenv('DYNAMIC_ADJUSTMENT_ENABLED', 'true').lower() == 'true'
POOL_ADJUSTMENT_COOLDOWN = 60  # Sekunden zwischen Pool-Anpassungen

# Timeout Configuration
GEMINI_REQUEST_TIMEOUT = 30  # seconds per Gemini request
BATCH_TIMEOUT_SECONDS = int(os.getenv('BATCH_TIMEOUT_SECONDS', '90'))  # 90 seconds per batch (realistischer)

class RequestStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"

@dataclass
class ClassificationResult:
    """Ergebnis einer LLM-Klassifikation"""
    speech_id: str
    is_brexit_related: bool
    confidence_score: float
    reasoning: str
    key_indicators: List[str]
    provider: str
    model: str
    processing_time: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None
    retry_count: int = 0

@dataclass
class ModelState:
    """Status eines Modells"""
    provider: str
    model_name: str
    requests_this_minute: int = 0
    requests_today: int = 0
    tokens_this_minute: int = 0
    last_minute_reset: datetime = field(default_factory=datetime.now)
    last_day_reset: datetime = field(default_factory=datetime.now)
    consecutive_errors: int = 0
    backoff_until: Optional[datetime] = None

@dataclass
class ProviderPool:
    """Provider-spezifischer Worker-Pool"""
    provider: str
    max_workers: int
    current_workers: int
    executor: Optional[ThreadPoolExecutor] = None
    active_requests: int = 0
    rpm_limit: int = 0
    health_score: float = 1.0
    last_adjustment: datetime = field(default_factory=datetime.now)
    consecutive_successes: int = 0
    consecutive_errors: int = 0

class CostTracker:
    """Verwaltet Kosten und Budget"""
    def __init__(self, max_budget: float):
        self.max_budget = max_budget
        self.total_cost = 0.0
        self.cost_by_provider = {}
        self.cost_by_model = {}
        self.lock = threading.Lock()
        self.budget_exceeded = False
        
    def add_cost(self, provider: str, model: str, cost: float) -> bool:
        """Fügt Kosten hinzu und prüft Budget"""
        with self.lock:
            if self.total_cost + cost > self.max_budget:
                self.budget_exceeded = True
                logger.warning(f"⚠️ Budget-Limit erreicht! Total: ${self.total_cost:.2f} + ${cost:.4f} > ${self.max_budget}")
                return False
            
            self.total_cost += cost
            self.cost_by_provider[provider] = self.cost_by_provider.get(provider, 0) + cost
            self.cost_by_model[model] = self.cost_by_model.get(model, 0) + cost
            
            # Warnung bei Schwellenwert
            if self.total_cost >= self.max_budget * COST_WARNING_THRESHOLD:
                logger.warning(f"⚠️ {COST_WARNING_THRESHOLD*100}% des Budgets erreicht: ${self.total_cost:.2f} / ${self.max_budget}")
            
            return True
    
    def get_remaining_budget(self) -> float:
        """Gibt verbleibendes Budget zurück"""
        with self.lock:
            return self.max_budget - self.total_cost
    
    def get_summary(self) -> Dict:
        """Gibt Kostenzusammenfassung zurück"""
        with self.lock:
            return {
                "total_cost": self.total_cost,
                "remaining_budget": self.max_budget - self.total_cost,
                "budget_used_percent": (self.total_cost / self.max_budget) * 100,
                "cost_by_provider": dict(self.cost_by_provider),
                "cost_by_model": dict(self.cost_by_model)
            }

class RateLimiter:
    """Intelligentes Rate Limiting mit Backoff und Provider Health Tracking"""
    def __init__(self):
        self.model_states = {}
        self.provider_health = {}  # Track Provider Health
        self.provider_pools = {}  # Provider-spezifische Pools
        self.lock = threading.Lock()
    
    def get_or_create_state(self, provider: str, model: str) -> ModelState:
        """Holt oder erstellt Model-State"""
        key = f"{provider}:{model}"
        if key not in self.model_states:
            self.model_states[key] = ModelState(provider=provider, model_name=model)
        return self.model_states[key]
    
    def can_make_request(self, provider: str, model: str, model_config: Dict, tokens: int) -> Tuple[bool, float]:
        """Prüft ob Request gemacht werden kann (OHNE Counter zu erhöhen)"""
        with self.lock:
            state = self.get_or_create_state(provider, model)
            now = datetime.now()
            
            # Prüfe Backoff
            if state.backoff_until and now < state.backoff_until:
                wait_time = (state.backoff_until - now).total_seconds()
                return False, wait_time
            
            # Reset Minute Counter
            if (now - state.last_minute_reset).total_seconds() >= 60:
                state.requests_this_minute = 0
                state.tokens_this_minute = 0
                state.last_minute_reset = now
            
            # Reset Day Counter
            if (now - state.last_day_reset).total_seconds() >= 86400:
                state.requests_today = 0
                state.last_day_reset = now
            
            # Prüfe Limits (OHNE Counter zu erhöhen!)
            rpm_limit = model_config.get("rpm_limit", float('inf'))
            tpm_limit = model_config.get("tpm_limit", float('inf'))
            rpd_limit = model_config.get("rpd_limit", float('inf'))
            
            if state.requests_this_minute >= rpm_limit:
                wait_time = 60 - (now - state.last_minute_reset).total_seconds()
                return False, max(0, wait_time)
            
            if state.tokens_this_minute + tokens >= tpm_limit:
                wait_time = 60 - (now - state.last_minute_reset).total_seconds()
                return False, max(0, wait_time)
            
            if rpd_limit and state.requests_today >= rpd_limit:
                wait_time = 86400 - (now - state.last_day_reset).total_seconds()
                return False, max(0, wait_time)
            
            # WICHTIG: Counter werden NICHT hier erhöht!
            # Das passiert erst nach erfolgreichem Request in register_success()
            return True, 0
    
    def register_error(self, provider: str, model: str, is_rate_limit: bool = False):
        """Registriert einen Fehler und setzt Backoff"""
        with self.lock:
            state = self.get_or_create_state(provider, model)
            state.consecutive_errors += 1
            
            # Update Provider Health
            if provider not in self.provider_health:
                self.provider_health[provider] = {"errors": 0, "successes": 0, "health_score": 1.0}
            
            self.provider_health[provider]["errors"] += 1
            self.provider_health[provider]["health_score"] = max(0.1, 
                self.provider_health[provider]["successes"] / 
                max(1, self.provider_health[provider]["errors"] + self.provider_health[provider]["successes"])
            )
            
            # Update Pool Health
            if provider in self.provider_pools:
                pool = self.provider_pools[provider]
                pool.health_score = self.provider_health[provider]["health_score"]
                pool.consecutive_errors += 1
                pool.consecutive_successes = 0
            
            # Pool-Anpassung bei Rate-Limits
            self.adjust_pool_on_rate_limit(provider, is_rate_limit)
            
            if is_rate_limit or state.consecutive_errors >= 2:  # Früherer Backoff
                # Sanfterer Backoff
                backoff_seconds = min(
                    INITIAL_RETRY_DELAY * (BACKOFF_FACTOR ** min(state.consecutive_errors, 3)),
                    MAX_RETRY_DELAY
                )
                state.backoff_until = datetime.now() + timedelta(seconds=backoff_seconds)
                logger.info(f"🔄 Backoff für {provider}:{model} - {backoff_seconds:.1f}s (Health: {self.provider_health[provider]['health_score']:.2f})")
    
    def register_success(self, provider: str, model: str, tokens: int = 0):
        """Registriert erfolgreichen Request und erhöht Counter"""
        with self.lock:
            state = self.get_or_create_state(provider, model)
            state.consecutive_errors = 0
            state.backoff_until = None
            
            # WICHTIG: Hier werden die Counter erhöht (nach erfolgreichem Request!)
            state.requests_this_minute += 1
            state.requests_today += 1
            state.tokens_this_minute += tokens
            
            # Update Provider Health
            if provider not in self.provider_health:
                self.provider_health[provider] = {"errors": 0, "successes": 0, "health_score": 1.0}
            
            self.provider_health[provider]["successes"] += 1
            self.provider_health[provider]["health_score"] = max(0.1, 
                self.provider_health[provider]["successes"] / 
                max(1, self.provider_health[provider]["errors"] + self.provider_health[provider]["successes"])
            )
            
            # Update Pool Health
            if provider in self.provider_pools:
                pool = self.provider_pools[provider]
                pool.health_score = self.provider_health[provider]["health_score"]
                pool.consecutive_successes += 1
                pool.consecutive_errors = 0
            
            # Pool-Anpassung bei Erfolgen
            self.adjust_pool_on_rate_limit(provider, False)
    
    def get_best_provider(self, available_models: List[Tuple[str, str]]) -> Tuple[str, str]:
        """Wählt den besten Provider basierend auf Health Score und Kosten"""
        with self.lock:
            if not available_models:
                return None, None
            
            # Sortiere nach Health Score und Kosten (höher = besser)
            scored_models = []
            for provider, model in available_models:
                health_score = self.provider_health.get(provider, {}).get("health_score", 1.0)
                
                # Zusätzliche Gewichtung: Günstigere Modelle bevorzugen
                # Kosten pro 1M Input Tokens (niedriger = besser)
                cost_weight = 0.0
                if provider == "gemini" and model == "gemini-1.5-flash-8b":
                    cost_weight = 0.1  # Günstigstes Modell
                elif provider == "gemini":
                    cost_weight = 0.05
                elif provider == "deepseek":
                    cost_weight = 0.02
                elif provider == "claude":
                    cost_weight = 0.01
                elif provider == "openai":
                    cost_weight = 0.0  # Teuerstes
                
                # Kombinierter Score: Health + Kosten-Bonus
                combined_score = health_score + cost_weight
                scored_models.append((combined_score, provider, model))
            
            # Sortiere absteigend nach kombiniertem Score
            scored_models.sort(reverse=True)
            
            # Debug: Zeige Auswahl
            if len(scored_models) > 1:
                print(f"🎯 Modell-Auswahl: {scored_models[0][1]}:{scored_models[0][2]} (Score: {scored_models[0][0]:.3f})")
            
            return scored_models[0][1], scored_models[0][2]  # Beste Provider:Model
    
    def calculate_optimal_workers(self, provider: str, rpm_limit: int) -> int:
        """Berechnet optimale Worker-Anzahl basierend auf RPM-Limit"""
        # DRAMATISCH mehr Worker für maximale Performance
        # Aggressive Parallelisierung für Geschwindigkeit
        
        # Spezielle Anpassungen für bekannte Provider (sehr aggressiv)
        if provider == "claude":
            optimal_workers = 8  # Claude: 8 Worker (50 RPM / 8 = ~6 RPM pro Worker)
        elif provider == "gemini":
            optimal_workers = 2  # Gemini: 2 Worker (10 RPM / 2 = 5 RPM pro Worker) - sehr niedrige Limits
        elif provider == "openai":
            optimal_workers = 20  # OpenAI: 20 Worker (500 RPM / 20 = 25 RPM pro Worker)
        elif provider == "deepseek":
            optimal_workers = 15  # DeepSeek: 15 Worker (100 RPM / 15 = ~7 RPM pro Worker)
        else:
            # Fallback: 1 Worker pro 10 RPM (sehr aggressiv)
            optimal_workers = max(5, min(25, rpm_limit // 10))
        
        return optimal_workers
    
    def create_provider_pools(self, available_models: List[Tuple[str, str]]) -> Dict[str, ProviderPool]:
        """Erstellt Provider-spezifische Pools basierend auf verfügbaren Modellen"""
        pools = {}
        
        for provider, model in available_models:
            if provider not in pools:
                # Finde RPM-Limit für diesen Provider
                rpm_limit = 50  # Default
                for config_provider, config in PROVIDERS_CONFIG.items():
                    if config_provider == provider:
                        for model_config in config["models"]:
                            if model_config["name"] == model:
                                rpm_limit = model_config.get("rpm_limit", 50)
                                break
                        break
                
                # Berechne optimale Worker-Anzahl
                max_workers = self.calculate_optimal_workers(provider, rpm_limit)
                
                # Erstelle Pool
                pool = ProviderPool(
                    provider=provider,
                    max_workers=max_workers,
                    current_workers=max_workers,
                    rpm_limit=rpm_limit,
                    health_score=1.0
                )
                
                # Erstelle ThreadPoolExecutor
                pool.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"{provider}-pool")
                pools[provider] = pool
                
                logger.info(f"🏊 Pool erstellt für {provider}: {max_workers} Worker (RPM: {rpm_limit})")
        
        return pools
    
    def adjust_pool_on_rate_limit(self, provider: str, is_rate_limit: bool):
        """Passt Pool-Größe basierend auf Rate-Limit-Feedback an"""
        if not DYNAMIC_ADJUSTMENT_ENABLED:
            return
        
        with self.lock:
            if provider not in self.provider_pools:
                return
            
            pool = self.provider_pools[provider]
            now = datetime.now()
            
            # Cooldown prüfen
            if (now - pool.last_adjustment).total_seconds() < POOL_ADJUSTMENT_COOLDOWN:
                return
            
            if is_rate_limit:
                # Reduziere Worker bei Rate-Limits
                pool.consecutive_errors += 1
                pool.consecutive_successes = 0
                
                if pool.current_workers > 1:
                    new_workers = max(1, pool.current_workers // 2)
                    pool.current_workers = new_workers
                    pool.last_adjustment = now
                    
                    logger.warning(f"🔧 Pool {provider} reduziert auf {new_workers} Worker (Rate-Limit)")
                    print(f"  🔧 {provider} Pool reduziert: {new_workers} Worker")
            else:
                # Erhöhe Worker bei Erfolgen
                pool.consecutive_successes += 1
                pool.consecutive_errors = 0
                
                if pool.consecutive_successes >= 5 and pool.current_workers < pool.max_workers:
                    new_workers = min(pool.max_workers, pool.current_workers + 1)
                    pool.current_workers = new_workers
                    pool.last_adjustment = now
                    
                    logger.info(f"🔧 Pool {provider} erhöht auf {new_workers} Worker (Success-Streak)")
                    print(f"  🔧 {provider} Pool erhöht: {new_workers} Worker")
    
    def get_best_available_pool(self, available_models: List[Tuple[str, str]]) -> Tuple[str, str]:
        """Wählt den besten verfügbaren Pool - AGRESSIVE SPEED-OPTIMIERUNG"""
        with self.lock:
            if not available_models:
                return None, None
            
            # AUSGEWOGENE SPEED-PRIORITÄT: Nutze alle Provider
            speed_priority = {
                "openai": 0.8,      # Hohe Priorität (schnell + hohe Limits)
                "deepseek": 0.7,    # Gute Priorität (schnell + moderate Limits)
                "claude": 0.6,      # Moderate Priorität (langsam aber stabil)
                "gemini": 0.5       # Niedrige Priorität (sehr langsam + niedrige Limits)
            }
            
            # Bewerte verfügbare Pools mit Speed-Priorität
            scored_pools = []
            for provider, model in available_models:
                if provider in self.provider_pools:
                    pool = self.provider_pools[provider]
                    
                    # Speed-Priorität (wichtigster Faktor)
                    speed_score = speed_priority.get(provider, 0.5)
                    
                    # Health Score (sekundär)
                    health_score = pool.health_score
                    
                    # Verfügbarkeit (tertiär)
                    availability_score = 1.0 - (pool.active_requests / max(pool.current_workers, 1))
                    
                    # Kombinierter Score: 70% Speed, 20% Health, 10% Availability
                    combined_score = speed_score * 0.7 + health_score * 0.2 + availability_score * 0.1
                    scored_pools.append((combined_score, provider, model))
            
            if not scored_pools:
                return available_models[0]  # Fallback
            
            # Sortiere nach Score (höchste zuerst)
            scored_pools.sort(reverse=True)
            return scored_pools[0][1], scored_pools[0][2]

class MultiLLMClassifier:
    """Multi-Provider LLM Klassifizierer mit Kostentracking"""
    
    def __init__(self, cost_tracker: CostTracker):
        self.providers = {}
        self.model_configs = {}
        self.api_keys = {}
        self.cost_tracker = cost_tracker
        self.rate_limiter = RateLimiter()
        self.load_api_keys()
        self.setup_providers()
        
    def load_api_keys(self):
        """Lädt API Keys aus .env Datei"""
        env_path = Path("/Users/julianbeese/Developer/Master/data_collection/.env")
        
        if not env_path.exists():
            logger.error("✗ .env Datei nicht gefunden!")
            return
        
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        self.api_keys[key] = value.strip().strip('"').strip("'")
            
            logger.info("✓ API Keys geladen")
        except Exception as e:
            logger.error(f"✗ Fehler beim Laden der .env Datei: {e}")
    
    def setup_providers(self):
        """Initialisiert alle Provider"""
        # Gemini
        if PROVIDERS_CONFIG["gemini"]["enabled"] and "GEMINI_API_KEY" in self.api_keys and GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=self.api_keys["GEMINI_API_KEY"])
                self.providers["gemini"] = {}
                for model_config in PROVIDERS_CONFIG["gemini"]["models"]:
                    model_name = model_config["name"]
                    self.providers["gemini"][model_name] = genai.GenerativeModel(model_name)
                    self.model_configs[f"gemini:{model_name}"] = model_config
                logger.info(f"✓ Gemini mit {len(PROVIDERS_CONFIG['gemini']['models'])} Modellen initialisiert")
                print(f"✅ Gemini initialisiert: {list(self.providers['gemini'].keys())}")
            except Exception as e:
                logger.error(f"✗ Gemini Initialisierung fehlgeschlagen: {e}")
                print(f"❌ Gemini Fehler: {e}")
        elif PROVIDERS_CONFIG["gemini"]["enabled"] and not GEMINI_AVAILABLE:
            logger.warning("⚠️ Gemini nicht verfügbar - google.generativeai nicht installiert")
            print("⚠️ Gemini nicht verfügbar - google.generativeai nicht installiert")
        else:
            print(f"⚠️ Gemini deaktiviert oder kein API Key - enabled: {PROVIDERS_CONFIG['gemini']['enabled']}, has_key: {'GEMINI_API_KEY' in self.api_keys}, available: {GEMINI_AVAILABLE}")
        
        # OpenAI
        if PROVIDERS_CONFIG["openai"]["enabled"] and "OPENAI_API_KEY" in self.api_keys and OPENAI_AVAILABLE:
            try:
                self.providers["openai"] = OpenAI(api_key=self.api_keys["OPENAI_API_KEY"])
                for model_config in PROVIDERS_CONFIG["openai"]["models"]:
                    self.model_configs[f"openai:{model_config['name']}"] = model_config
                logger.info(f"✓ OpenAI mit {len(PROVIDERS_CONFIG['openai']['models'])} Modellen initialisiert")
                print(f"✅ OpenAI initialisiert: {len(PROVIDERS_CONFIG['openai']['models'])} Modelle")
            except Exception as e:
                logger.error(f"✗ OpenAI Initialisierung fehlgeschlagen: {e}")
                print(f"❌ OpenAI Fehler: {e}")
        elif PROVIDERS_CONFIG["openai"]["enabled"] and not OPENAI_AVAILABLE:
            logger.warning("⚠️ OpenAI nicht verfügbar - openai nicht installiert")
            print("⚠️ OpenAI nicht verfügbar - openai nicht installiert")
        else:
            print(f"⚠️ OpenAI deaktiviert oder kein API Key - enabled: {PROVIDERS_CONFIG['openai']['enabled']}, has_key: {'OPENAI_API_KEY' in self.api_keys}, available: {OPENAI_AVAILABLE}")
        
        # Claude
        if PROVIDERS_CONFIG["claude"]["enabled"] and "ANTHROPIC_API_KEY" in self.api_keys and ANTHROPIC_AVAILABLE:
            try:
                self.providers["claude"] = Anthropic(api_key=self.api_keys["ANTHROPIC_API_KEY"])
                for model_config in PROVIDERS_CONFIG["claude"]["models"]:
                    self.model_configs[f"claude:{model_config['name']}"] = model_config
                logger.info(f"✓ Claude mit {len(PROVIDERS_CONFIG['claude']['models'])} Modellen initialisiert")
                print(f"✅ Claude initialisiert: {len(PROVIDERS_CONFIG['claude']['models'])} Modelle")
            except Exception as e:
                logger.error(f"✗ Claude Initialisierung fehlgeschlagen: {e}")
                print(f"❌ Claude Fehler: {e}")
        elif PROVIDERS_CONFIG["claude"]["enabled"] and not ANTHROPIC_AVAILABLE:
            logger.warning("⚠️ Claude nicht verfügbar - anthropic nicht installiert")
            print("⚠️ Claude nicht verfügbar - anthropic nicht installiert")
        else:
            print(f"⚠️ Claude deaktiviert oder kein API Key - enabled: {PROVIDERS_CONFIG['claude']['enabled']}, has_key: {'ANTHROPIC_API_KEY' in self.api_keys}, available: {ANTHROPIC_AVAILABLE}")
        
        # DeepSeek
        if PROVIDERS_CONFIG["deepseek"]["enabled"] and "DEEPSEEK_API_KEY" in self.api_keys and OPENAI_AVAILABLE:
            try:
                self.providers["deepseek"] = OpenAI(
                    api_key=self.api_keys["DEEPSEEK_API_KEY"],
                    base_url=PROVIDERS_CONFIG["deepseek"]["base_url"]
                )
                for model_config in PROVIDERS_CONFIG["deepseek"]["models"]:
                    self.model_configs[f"deepseek:{model_config['name']}"] = model_config
                logger.info(f"✓ DeepSeek mit {len(PROVIDERS_CONFIG['deepseek']['models'])} Modellen initialisiert")
                print(f"✅ DeepSeek initialisiert: {len(PROVIDERS_CONFIG['deepseek']['models'])} Modelle")
            except Exception as e:
                logger.error(f"✗ DeepSeek Initialisierung fehlgeschlagen: {e}")
                print(f"❌ DeepSeek Fehler: {e}")
        elif PROVIDERS_CONFIG["deepseek"]["enabled"] and not OPENAI_AVAILABLE:
            logger.warning("⚠️ DeepSeek nicht verfügbar - openai nicht installiert")
            print("⚠️ DeepSeek nicht verfügbar - openai nicht installiert")
        else:
            print(f"⚠️ DeepSeek deaktiviert oder kein API Key - enabled: {PROVIDERS_CONFIG['deepseek']['enabled']}, has_key: {'DEEPSEEK_API_KEY' in self.api_keys}, available: {OPENAI_AVAILABLE}")
        
        # Mistral
        if PROVIDERS_CONFIG["mistral"]["enabled"] and "MISTRAL_API_KEY" in self.api_keys:
            try:
                from mistralai import Mistral
                self.providers["mistral"] = Mistral(api_key=self.api_keys["MISTRAL_API_KEY"])
                for model_config in PROVIDERS_CONFIG["mistral"]["models"]:
                    self.model_configs[f"mistral:{model_config['name']}"] = model_config
                logger.info(f"✓ Mistral mit {len(PROVIDERS_CONFIG['mistral']['models'])} Modellen initialisiert")
            except Exception as e:
                logger.error(f"✗ Mistral Initialisierung fehlgeschlagen: {e}")
        
        # Together
        if PROVIDERS_CONFIG["together"]["enabled"] and "TOGETHER_API_KEY" in self.api_keys:
            try:
                self.providers["together"] = OpenAI(
                    api_key=self.api_keys["TOGETHER_API_KEY"],
                    base_url=PROVIDERS_CONFIG["together"]["base_url"]
                )
                for model_config in PROVIDERS_CONFIG["together"]["models"]:
                    self.model_configs[f"together:{model_config['name']}"] = model_config
                logger.info(f"✓ Together AI mit {len(PROVIDERS_CONFIG['together']['models'])} Modellen initialisiert")
            except Exception as e:
                logger.error(f"✗ Together Initialisierung fehlgeschlagen: {e}")
        
        # Debug: Zeige initialisierte Provider
        print(f"\n🔍 INITIALISIERTE PROVIDER:")
        for provider_name, provider_obj in self.providers.items():
            if isinstance(provider_obj, dict):
                print(f"  ✅ {provider_name}: {list(provider_obj.keys())}")
            else:
                print(f"  ✅ {provider_name}: {type(provider_obj).__name__}")
        
        if not self.providers:
            print("  ❌ Keine Provider initialisiert!")
        else:
            print(f"  📊 Gesamt: {len(self.providers)} Provider verfügbar")
    
    def estimate_tokens(self, text: str) -> int:
        """Schätzt Token-Anzahl (4 Zeichen = 1 Token)"""
        return max(1, len(text) // 4)
    
    def calculate_cost(self, input_tokens: int, output_tokens: int, model_config: Dict) -> float:
        """Berechnet Kosten für einen Request"""
        input_cost = (input_tokens / 1_000_000) * model_config["cost_per_1M_input"]
        output_cost = (output_tokens / 1_000_000) * model_config["cost_per_1M_output"]
        return input_cost + output_cost
    
    def create_prompt(self, speech_text: str, speaker_name: str, debate_title: str, date: str) -> str:
        """Erstellt den optimierten Analyse-Prompt"""
        # Intelligente Text-Kürzung basierend auf Länge
        if len(speech_text) > 1500:
            # Versuche den wichtigsten Teil zu finden (erste 800 + letzte 200 Zeichen)
            speech_excerpt = speech_text[:800] + "...[middle truncated]..." + speech_text[-200:]
        else:
            speech_excerpt = speech_text
        
        # Kürze Context für bessere Performance
        context = f"{speaker_name}, {date}"
        if debate_title and len(debate_title) > 50:
            context += f", {debate_title[:50]}..."
        
        return f"""You are a Brexit expert, and have to determine if this UK Parliament speech about Brexit.
        Please provide the confidence score for your answer. (0 = no relation to Brexit, 1 = very likely relation to Brexit)

Context: {context}
Speech: {speech_excerpt}

JSON only:
{{"is_brexit_related": true/false, "confidence_score": 0.0-1.0, "reasoning": "brief", "key_indicators": ["max 2"]}}"""
    
    def classify_with_retry(self, provider: str, model: str, speech_data: Dict, retry_count: int = 0) -> ClassificationResult:
        """Klassifiziert mit automatischem Retry bei Fehlern"""
        start_time = time.time()
        model_config = self.model_configs.get(f"{provider}:{model}")
        
        if not model_config:
            return ClassificationResult(
                speech_id=speech_data["speech_id"],
                is_brexit_related=False,
                confidence_score=0.0,
                reasoning="Model config not found",
                key_indicators=[],
                provider=provider,
                model=model,
                processing_time=time.time() - start_time,
                error="Model config not found",
                retry_count=retry_count
            )
        
        # Erstelle Prompt
        prompt = self.create_prompt(
            speech_data["speech_text"],
            speech_data["speaker_name"],
            speech_data["debate_title"],
            speech_data["date"]
        )
        
        # Schätze Tokens
        estimated_input_tokens = self.estimate_tokens(prompt)
        estimated_output_tokens = 100  # Geschätzte Antwortlänge
        
        # Schätze Kosten
        estimated_cost = self.calculate_cost(
            estimated_input_tokens, 
            estimated_output_tokens, 
            model_config
        )
        
        # Prüfe Budget VOR dem Request
        if not self.cost_tracker.add_cost(provider, model, estimated_cost):
            return ClassificationResult(
                speech_id=speech_data["speech_id"],
                is_brexit_related=False,
                confidence_score=0.0,
                reasoning="Budget exceeded",
                key_indicators=[],
                provider=provider,
                model=model,
                processing_time=time.time() - start_time,
                error="Budget limit reached",
                retry_count=retry_count
            )
        
        # Prüfe Rate Limits
        can_proceed, wait_time = self.rate_limiter.can_make_request(
            provider, model, model_config, estimated_input_tokens
        )
        
        if not can_proceed:
            if retry_count < MAX_RETRIES:
                logger.info(f"⏳ Rate limit für {provider}:{model}, warte {wait_time:.1f}s...")
                time.sleep(wait_time + random.uniform(0, 1))  # Kleine zufällige Verzögerung
                return self.classify_with_retry(provider, model, speech_data, retry_count + 1)
            else:
                # Kosten zurückerstatten bei Fehlschlag
                self.cost_tracker.add_cost(provider, model, -estimated_cost)
                return ClassificationResult(
                    speech_id=speech_data["speech_id"],
                    is_brexit_related=False,
                    confidence_score=0.0,
                    reasoning="Rate limit exceeded after retries",
                    key_indicators=[],
                    provider=provider,
                    model=model,
                    processing_time=time.time() - start_time,
                    error="Rate limit exceeded",
                    retry_count=retry_count
                )
        
        try:
            # Provider-spezifische Implementierung
            logger.info(f"🚀 Starting API call: {provider}:{model}")
            if provider == "gemini":
                # Add request options with timeout
                from google.generativeai.types import RequestOptions
                
                request_options = RequestOptions(
                    timeout=GEMINI_REQUEST_TIMEOUT  # Use configured timeout
                )
                
                def gemini_call():
                    return self.providers["gemini"][model].generate_content(
                        prompt,
                        request_options=request_options
                    )
                
                response = with_timeout(gemini_call, 30)  # 30-second timeout
                result = self._parse_response(response.text)
                actual_input_tokens = estimated_input_tokens
                actual_output_tokens = self.estimate_tokens(response.text)

            elif provider == "openai":
                def openai_call():
                    # GPT-5 Modelle verwenden max_completion_tokens statt max_tokens
                    if "gpt-5" in model:
                        return self.providers["openai"].chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                            max_completion_tokens=150  
                        )
                    else:
                        # Ältere Modelle (gpt-4o-mini, gpt-3.5-turbo) verwenden max_tokens
                        return self.providers["openai"].chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=150 
                        )

                response = with_timeout(openai_call, 30)  # 30-second timeout
                result = self._parse_response(response.choices[0].message.content)
                actual_input_tokens = response.usage.prompt_tokens if response.usage else estimated_input_tokens
                actual_output_tokens = response.usage.completion_tokens if response.usage else estimated_output_tokens
            elif provider == "claude":
                def claude_call():
                    return self.providers["claude"].messages.create(
                        model=model,
                        max_tokens=150,
                        temperature=0.1,
                        messages=[{"role": "user", "content": prompt}]
                    )
                
                response = with_timeout(claude_call, 30)  # 30-second timeout
                result = self._parse_response(response.content[0].text)
                actual_input_tokens = response.usage.input_tokens if hasattr(response, 'usage') else estimated_input_tokens
                actual_output_tokens = response.usage.output_tokens if hasattr(response, 'usage') else estimated_output_tokens
                
            elif provider in ["deepseek", "together"]:
                def deepseek_call():
                    return self.providers[provider].chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=150
                    )
                
                response = with_timeout(deepseek_call, 30)  # 30-second timeout
                result = self._parse_response(response.choices[0].message.content)
                actual_input_tokens = response.usage.prompt_tokens if response.usage else estimated_input_tokens
                actual_output_tokens = response.usage.completion_tokens if response.usage else estimated_output_tokens
                
            elif provider == "mistral":
                def mistral_call():
                    return self.providers["mistral"].chat.complete(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=150
                    )
                
                response = with_timeout(mistral_call, 30)  # 30-second timeout
                result = self._parse_response(response.choices[0].message.content)
                actual_input_tokens = response.usage.prompt_tokens if response.usage else estimated_input_tokens
                actual_output_tokens = response.usage.completion_tokens if response.usage else estimated_output_tokens
            else:
                raise ValueError(f"Unbekannter Provider: {provider}")
            
            # Berechne tatsächliche Kosten
            actual_cost = self.calculate_cost(actual_input_tokens, actual_output_tokens, model_config)
            
            # Korrigiere Kostendifferenz
            cost_diff = actual_cost - estimated_cost
            if abs(cost_diff) > 0.0001:  # Nur bei signifikanter Differenz
                self.cost_tracker.add_cost(provider, model, cost_diff)
            
            # Debug: Zeige das Ergebnis
            logger.info(f"✅ API call successful: {provider}:{model} in {time.time() - start_time:.2f}s")
            logger.info(f"🔍 Result: {result}")
            self.rate_limiter.register_success(provider, model, actual_input_tokens)
            
            return ClassificationResult(
                speech_id=speech_data["speech_id"],
                is_brexit_related=result["is_brexit_related"],
                confidence_score=result["confidence_score"],
                reasoning=result["reasoning"],
                key_indicators=result["key_indicators"],
                provider=provider,
                model=model,
                processing_time=time.time() - start_time,
                input_tokens=actual_input_tokens,
                output_tokens=actual_output_tokens,
                cost_usd=actual_cost,
                retry_count=retry_count
            )
            
        except TimeoutError as e:
            logger.warning(f"⏱️ Timeout bei {provider}:{model}: {str(e)}")
            self.rate_limiter.register_error(provider, model, is_rate_limit=False)
            
            # Retry with different provider
            available_models = self.get_available_models()
            for alt_provider, alt_model in available_models:
                if alt_provider != provider:
                    logger.info(f"🔄 Timeout-Retry mit {alt_provider}:{alt_model}")
                    return self.classify_with_retry(alt_provider, alt_model, speech_data, retry_count + 1)
            
            # No alternative found
            return ClassificationResult(
                speech_id=speech_data["speech_id"],
                is_brexit_related=False,
                confidence_score=0.0,
                reasoning="Timeout - no alternative provider available",
                key_indicators=[],
                provider=provider,
                model=model,
                processing_time=time.time() - start_time,
                error="Timeout - no alternative provider available",
                retry_count=retry_count
            )
            
        except Exception as e:
            error_msg = str(e)
            is_rate_limit = "rate" in error_msg.lower() or "429" in error_msg or "quota" in error_msg.lower()
            is_model_error = "404" in error_msg or "not found" in error_msg.lower() or "not supported" in error_msg.lower()
            
            # Registriere Fehler
            self.rate_limiter.register_error(provider, model, is_rate_limit)
            
            # Bei Model-Fehlern: Sofortiger Fallback zu anderem Provider
            if is_model_error:
                logger.error(f"❌ Model-Fehler bei {provider}:{model}: {error_msg[:100]}")
                # Versuche mit anderem Provider
                available_models = self.get_available_models()
                for alt_provider, alt_model in available_models:
                    if alt_provider != provider and alt_provider in self.providers:
                        # Prüfe ob das Modell tatsächlich verfügbar ist
                        if alt_provider == "gemini" and alt_model in self.providers["gemini"]:
                            logger.info(f"🔄 Fallback zu {alt_provider}:{alt_model}")
                            return self.classify_with_retry(alt_provider, alt_model, speech_data, retry_count)
                        elif alt_provider != "gemini":
                            logger.info(f"🔄 Fallback zu {alt_provider}:{alt_model}")
                            return self.classify_with_retry(alt_provider, alt_model, speech_data, retry_count)
            
            # Retry bei Rate Limit Errors (nur bei gesunden Providern)
            if is_rate_limit and retry_count < MAX_RETRIES:
                health_score = self.rate_limiter.provider_health.get(provider, {}).get("health_score", 1.0)
                if health_score > 0.3:  # Nur bei gesunden Providern retry
                    wait_time = INITIAL_RETRY_DELAY * (BACKOFF_FACTOR ** retry_count)
                    logger.warning(f"🔄 Rate limit error für {provider}:{model}, Retry {retry_count+1}/{MAX_RETRIES} in {wait_time:.1f}s")
                    time.sleep(wait_time)
                    return self.classify_with_retry(provider, model, speech_data, retry_count + 1)
                else:
                    logger.warning(f"⚠️ Provider {provider} zu ungesund für Retry (Health: {health_score:.2f})")
            
            # Kosten zurückerstatten bei Fehler
            self.cost_tracker.add_cost(provider, model, -estimated_cost)
            
            logger.error(f"❌ Fehler bei {provider}:{model} nach {retry_count} Retries: {error_msg}")
            
            return ClassificationResult(
                speech_id=speech_data["speech_id"],
                is_brexit_related=False,
                confidence_score=0.0,
                reasoning="Classification failed",
                key_indicators=[],
                provider=provider,
                model=model,
                processing_time=time.time() - start_time,
                error=error_msg[:200],
                retry_count=retry_count
            )
    
    def _parse_response(self, response_text: str) -> Dict:
        """Parsed die JSON-Antwort"""
        try:
            # Bereinige Response
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            result = json.loads(response_text.strip())
            
            return {
                "is_brexit_related": bool(result.get("is_brexit_related", False)),
                "confidence_score": min(1.0, max(0.0, float(result.get("confidence_score", 0.0)))),
                "reasoning": str(result.get("reasoning", ""))[:100],
                "key_indicators": list(result.get("key_indicators", []))[:3]
            }
        except Exception as e:
            logger.error(f"Parse error: {e}, Response: {response_text[:200]}")
            # Fallback
            return {
                "is_brexit_related": False,
                "confidence_score": 0.0,
                "reasoning": "Parse error",
                "key_indicators": []
            }
    
    def get_available_models(self) -> List[Tuple[str, str]]:
        """Gibt Liste aller verfügbaren Modelle zurück"""
        models = []
        for provider, config in PROVIDERS_CONFIG.items():
            if config["enabled"] and provider in self.providers:
                for model_config in config["models"]:
                    # Prüfe ob das Modell tatsächlich verfügbar ist
                    if provider == "gemini" and model_config["name"] in self.providers["gemini"]:
                        models.append((provider, model_config["name"]))
                    elif provider != "gemini":
                        models.append((provider, model_config["name"]))
        return models

def copy_database_with_new_columns():
    """Erstellt eine Kopie der Datenbank mit neuen Spalten"""
    print("\n" + "=" * 80)
    print("MULTI-LLM BREXIT-KLASSIFIKATION MIT KOSTENTRACKING")
    print("=" * 80)
    print(f"Budget-Limit: ${MAX_BUDGET_USD:.2f}")
    print("=" * 80)
    
    if not Path(INPUT_DB).exists():
        logger.error(f"✗ Input-Datenbank {INPUT_DB} nicht gefunden!")
        return None
    
    logger.info(f"✓ Input-Datenbank gefunden: {INPUT_DB}")
    
    # Backup falls Output DB existiert
    if Path(OUTPUT_DB).exists():
        backup_path = f"{OUTPUT_DB}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.rename(OUTPUT_DB, backup_path)
        logger.info(f"✓ Alte Datenbank gesichert als: {backup_path}")
    
    logger.info("Kopiere Datenbank...")
    conn_source = duckdb.connect(INPUT_DB, read_only=True)
    conn_target = duckdb.connect(OUTPUT_DB)
    
    try:
        # Kopiere alle Tabellen
        tables = ['debates', 'topics', 'speeches']
        
        for table_name in tables:
            logger.info(f"  Kopiere {table_name}...")
            
            # Hole Schema
            schema_info = conn_source.execute(f"PRAGMA table_info('{table_name}')").fetchall()
            columns = []
            for row in schema_info:
                col_name = row[1]
                col_type = row[2]
                columns.append(f"{col_name} {col_type}")
            
            # Erstelle Tabelle
            create_sql = f"CREATE TABLE {table_name} (\n    " + ",\n    ".join(columns) + "\n)"
            conn_target.execute(create_sql)
            
            # Kopiere Daten
            data = conn_source.execute(f"SELECT * FROM {table_name}").fetchall()
            if data:
                placeholders = ", ".join(["?" for _ in columns])
                insert_sql = f"INSERT INTO {table_name} VALUES ({placeholders})"
                
                for row in data:
                    conn_target.execute(insert_sql, row)
                
                logger.info(f"    ✓ {len(data):,} Zeilen kopiert")
        
        # Füge neue Spalten hinzu
        logger.info("Füge LLM-Spalten hinzu...")
        conn_target.execute("ALTER TABLE speeches ADD COLUMN llm_classified_brexit BOOLEAN")
        conn_target.execute("ALTER TABLE speeches ADD COLUMN llm_confidence_score FLOAT")
        conn_target.execute("ALTER TABLE speeches ADD COLUMN llm_reasoning VARCHAR")
        conn_target.execute("ALTER TABLE speeches ADD COLUMN llm_key_indicators VARCHAR")
        conn_target.execute("ALTER TABLE speeches ADD COLUMN llm_provider VARCHAR")
        conn_target.execute("ALTER TABLE speeches ADD COLUMN llm_model VARCHAR")
        conn_target.execute("ALTER TABLE speeches ADD COLUMN llm_processing_time FLOAT")
        conn_target.execute("ALTER TABLE speeches ADD COLUMN llm_input_tokens INTEGER")
        conn_target.execute("ALTER TABLE speeches ADD COLUMN llm_output_tokens INTEGER")
        conn_target.execute("ALTER TABLE speeches ADD COLUMN llm_cost_usd FLOAT")
        conn_target.execute("ALTER TABLE speeches ADD COLUMN llm_processed BOOLEAN DEFAULT FALSE")
        conn_target.execute("ALTER TABLE speeches ADD COLUMN llm_error VARCHAR")
        conn_target.execute("ALTER TABLE speeches ADD COLUMN llm_retry_count INTEGER DEFAULT 0")
        conn_target.execute("ALTER TABLE speeches ADD COLUMN llm_processed_at TIMESTAMP")
        
        logger.info("✓ LLM-Spalten hinzugefügt")
        
        # Erstelle Index für Performance
        conn_target.execute("CREATE INDEX idx_llm_processed ON speeches(llm_processed)")
        conn_target.execute("CREATE INDEX idx_llm_brexit ON speeches(llm_classified_brexit)")
        
        return conn_target
        
    finally:
        conn_source.close()

def process_speeches_with_budget(conn):
    """Verarbeitet Speeches mit Budget-Kontrolle und robuster Fehlerbehandlung"""
    logger.info("\nStarte Multi-LLM-Klassifikation mit Kostentracking...")
    print(f"\n🔧 INITIALISIERE KLASSIFIKATIONSSYSTEM")
    
    # Test-Modus prüfen
    test_mode = os.getenv('TEST_MODE', 'false').lower() == 'true'
    test_limit = int(os.getenv('TEST_LIMIT', '500'))
    
    if test_mode:
        print(f"🧪 TEST-MODUS AKTIVIERT - Verarbeite maximal {test_limit} Speeches")
        logger.info(f"TEST-MODUS: Verarbeite maximal {test_limit} Speeches")
    
    # Initialisiere Komponenten
    print(f"  💰 Erstelle CostTracker mit Budget ${MAX_BUDGET_USD}")
    cost_tracker = CostTracker(MAX_BUDGET_USD)
    print(f"  🤖 Erstelle MultiLLMClassifier")
    classifier = MultiLLMClassifier(cost_tracker)
    
    print(f"  🔍 Prüfe verfügbare Modelle...")
    available_models = classifier.get_available_models()
    if not available_models:
        logger.error("✗ Keine Modelle verfügbar!")
        print(f"❌ FEHLER: Keine Modelle verfügbar!")
        return
    
    print(f"✅ {len(available_models)} Modelle verfügbar:")
    logger.info(f"✓ {len(available_models)} Modelle verfügbar:")
    for provider, model in available_models:
        config = classifier.model_configs.get(f"{provider}:{model}")
        if config:
            print(f"    • {provider}:{model} - ${config['cost_per_1M_input']:.3f}/1M input")
            logger.info(f"  • {provider}:{model} - ${config['cost_per_1M_input']:.3f}/1M input")
    
    # Hole Statistiken
    print(f"\n📊 DATENBANK-STATISTIKEN:")
    total_speeches = conn.execute("SELECT COUNT(*) FROM speeches").fetchone()[0]
    processed_count = conn.execute("SELECT COUNT(*) FROM speeches WHERE llm_processed = TRUE").fetchone()[0]
    
    print(f"  Gesamt Speeches: {total_speeches:,}")
    print(f"  Bereits verarbeitet: {processed_count:,}")
    print(f"  Noch zu verarbeiten: {total_speeches - processed_count:,}")
    print(f"  Verbleibendes Budget: ${cost_tracker.get_remaining_budget():.2f}")
    
    logger.info(f"\nZu verarbeitende Speeches: {total_speeches:,}")
    logger.info(f"Bereits verarbeitet: {processed_count:,}")
    logger.info(f"Verbleibendes Budget: ${cost_tracker.get_remaining_budget():.2f}")
    
    if processed_count >= total_speeches:
        logger.info("✓ Alle Speeches wurden bereits verarbeitet!")
        print(f"✅ Alle Speeches wurden bereits verarbeitet!")
        return
    
    # Processing Loop
    print(f"\n🚀 STARTE BATCH-VERARBEITUNG")
    batch_num = 0
    total_processed = processed_count
    brexit_count = conn.execute("SELECT COUNT(*) FROM speeches WHERE llm_classified_brexit = TRUE").fetchone()[0]
    errors_count = 0
    start_time = time.time()
    
    print(f"  Startzeit: {datetime.now().strftime('%H:%M:%S')}")
    print(f"  Batch-Größe: {BATCH_SIZE}")
    
    # Erstelle Provider-spezifische Pools
    print(f"  🏊 Erstelle Provider-Pools...")
    classifier.rate_limiter.provider_pools = classifier.rate_limiter.create_provider_pools(available_models)
    
    # Zeige Pool-Konfiguration
    print(f"  📊 Pool-Konfiguration:")
    for provider, pool in classifier.rate_limiter.provider_pools.items():
        print(f"    • {provider}: {pool.current_workers} Worker (RPM: {pool.rpm_limit})")
    
    # Intelligente Model Selection basierend auf Health Score
    model_selection_strategy = "pool_based"  # Neue Strategie für Pool-Management
    
    try:
        while not cost_tracker.budget_exceeded:
            # Test-Modus: Prüfe ob Limit erreicht (nur neue Verarbeitungen zählen)
            if test_mode and (total_processed - processed_count) >= test_limit:
                print(f"🧪 TEST-MODUS: Limit von {test_limit} neuen Speeches erreicht")
                logger.info(f"TEST-MODUS: Limit von {test_limit} neuen Speeches erreicht")
                break
            
            # Hole nächsten Batch
            print(f"\n🔍 Lade nächsten Batch...")
            batch_speeches = conn.execute(f"""
                SELECT s.speech_id, s.speech_text, s.speaker_name, 
                       d.major_heading_text, d.date
                FROM speeches s
                JOIN debates d ON s.debate_id = d.debate_id
                WHERE s.llm_processed = FALSE
                LIMIT {BATCH_SIZE}
            """).fetchall()
            
            if not batch_speeches:
                logger.info("✓ Keine weiteren Speeches zu verarbeiten.")
                print(f"✅ Keine weiteren Speeches zu verarbeiten.")
                break
            
            batch_num += 1
            print(f"\n📦 BATCH {batch_num} ({len(batch_speeches)} Speeches)")
            print(f"  Budget verwendet: ${cost_tracker.total_cost:.2f} / ${MAX_BUDGET_USD:.2f}")
            print(f"  Verbleibendes Budget: ${cost_tracker.get_remaining_budget():.2f}")
            
            logger.info(f"\n--- Batch {batch_num} ({len(batch_speeches)} Speeches) ---")
            logger.info(f"Budget verwendet: ${cost_tracker.total_cost:.2f} / ${MAX_BUDGET_USD:.2f}")
            
            # Erstelle Speech-Daten
            print(f"  📝 Bereite {len(batch_speeches)} Speeches vor...")
            speech_data_list = []
            for speech_id, speech_text, speaker_name, debate_title, date in batch_speeches:
                # Kürze Text für Kosteneffizienz
                original_length = len(speech_text)
                if len(speech_text) > 3000:
                    speech_text = speech_text[:3000] + "..."
                
                speech_data_list.append({
                    "speech_id": speech_id,
                    "speech_text": speech_text,
                    "speaker_name": speaker_name or "Unknown",
                    "debate_title": debate_title or "Unknown",
                    "date": str(date) if date else "Unknown"
                })
            
            print(f"  ✅ {len(speech_data_list)} Speeches vorbereitet")
            if any(len(s['speech_text']) > 3000 for s in speech_data_list):
                print(f"  ⚠️  Einige Speeches wurden gekürzt (max 3000 Zeichen)")
            
            # AGRESSIVE BATCH-VERARBEITUNG für maximale Geschwindigkeit
            print(f"  🚀 Starte AGRESSIVE parallele Verarbeitung...")
            
            # Sammle alle Futures von allen Pools
            all_futures = []
            model_index = 0
            
            # SPEED-OPTIMIERUNG: Submit alle Tasks sofort ohne Wartezeiten
            for i, speech_data in enumerate(speech_data_list):
                if cost_tracker.budget_exceeded:
                    print(f"  ⚠️  Budget-Limit erreicht, stoppe weitere Tasks")
                    break
                
                # HYBRID-STRATEGIE: Speed-Priorität + Round-Robin für Ausgewogenheit
                if model_selection_strategy == "pool_based":
                    # 70% Speed-Priorität, 30% Round-Robin für Ausgewogenheit
                    if i % 10 < 7:  # 70% der Requests mit Speed-Priorität
                        provider, model = classifier.rate_limiter.get_best_available_pool(available_models)
                        if provider is None:  # Fallback zu Round-Robin
                            provider, model = available_models[model_index % len(available_models)]
                            model_index += 1
                    else:  # 30% der Requests mit Round-Robin
                        provider, model = available_models[model_index % len(available_models)]
                        model_index += 1
                else:
                    # Round-Robin: Verteile gleichmäßig auf alle Provider
                    provider, model = available_models[model_index % len(available_models)]
                    model_index += 1
                
                # Prüfe ob Provider und Modell verfügbar sind
                if provider not in classifier.providers:
                    # Fallback zu verfügbarem Provider
                    for alt_provider, alt_model in available_models:
                        if alt_provider in classifier.providers:
                            provider, model = alt_provider, alt_model
                            break
                
                # Prüfe ob das Modell tatsächlich verfügbar ist
                if provider == "gemini" and model not in classifier.providers["gemini"]:
                    # Fallback zu verfügbarem Modell
                    for alt_provider, alt_model in available_models:
                        if alt_provider in classifier.providers:
                            if alt_provider == "gemini" and alt_model in classifier.providers["gemini"]:
                                provider, model = alt_provider, alt_model
                                break
                            elif alt_provider != "gemini":
                                provider, model = alt_provider, alt_model
                                break
                
                # Prüfe ob Pool verfügbar ist
                if provider not in classifier.rate_limiter.provider_pools:
                    print(f"    ⚠️  Pool für {provider} nicht verfügbar, überspringe")
                    continue
                
                pool = classifier.rate_limiter.provider_pools[provider]
                
                # SPEED-OPTIMIERUNG: Keine Pool-Kapazitätsprüfung - submit sofort!
                strategy = "SPEED" if i % 10 < 7 else "ROUND-ROBIN"
                print(f"    📤 Speech {i+1}/{len(speech_data_list)} → {provider}:{model} ({strategy})")
                
                # Submit Task an Provider-spezifischen Pool (DIREKT!)
                # Verwende classify_with_retry direkt ohne Wrapper
                future = pool.executor.submit(
                    classifier.classify_with_retry,
                    provider, model, speech_data
                )
                all_futures.append(future)
                
                # Update Pool-Status
                pool.active_requests += 1

            # Sammle Ergebnisse von allen Pools mit einfacher Timeout-Logik
            print(f"  ⏳ Warte auf Ergebnisse...")
            completed_count = 0
            batch_start_time = time.time()
            batch_results = []  # Store results for batch processing

            # Process futures as they complete - ULTRA SIMPLE AND ROBUST
            print(f"  🔍 Sammle verfügbare Ergebnisse...")
            print(f"  ⏳ Warte auf {len(all_futures)} Futures...")
            
            # IMPROVED: Use as_completed for better future handling
            start_wait_time = time.time()
            max_wait_time = BATCH_TIMEOUT_SECONDS
            
            print(f"  🔍 DEBUG: Starte Future-Verarbeitung mit {len(all_futures)} Futures")
            
            # Use as_completed for better handling
            from concurrent.futures import as_completed
            
            # SIMPLIFIED: Process futures without timeout - let them complete naturally
            print(f"  🔍 DEBUG: Verarbeite {len(all_futures)} Futures ohne Timeout...")
            print(f"  🔍 DEBUG: Futures Status: {[f.done() for f in all_futures[:5]]}... (erste 5)")
            
            # Process futures as they complete - NO TIMEOUT
            for i, future in enumerate(as_completed(all_futures)):
                try:
                    print(f"  🔍 DEBUG: Verarbeite Future {i+1}/{len(all_futures)}")
                    result = future.result(timeout=5.0)  # 5 second timeout per individual future
                    completed_count += 1
                    batch_results.append(result)  # Store result
                    
                    print(f"    ✅ Speech {completed_count}/{len(speech_data_list)} verarbeitet")
                    print(f"    🔍 DEBUG: Result für speech_id={result.speech_id}: brexit={result.is_brexit_related}, provider={result.provider}")

                    # Update Pool-Status
                    if result.provider in classifier.rate_limiter.provider_pools:
                        pool = classifier.rate_limiter.provider_pools[result.provider]
                        pool.active_requests = max(0, pool.active_requests - 1)

                    # Register success with rate limiter
                    classifier.rate_limiter.register_success(
                        result.provider,
                        result.model,
                        result.input_tokens
                    )

                    # Log progress
                    if result.is_brexit_related:
                        print(f"      🎯 Brexit-Speech erkannt! (Confidence: {result.confidence_score:.2f})")
                    if result.error:
                        errors_count += 1
                        print(f"      ❌ Fehler: {result.error[:100]}")

                except Exception as e:
                    logger.error(f"Error processing future result: {e}")
                    errors_count += 1
                    print(f"      ❌ Future processing error: {e}")
                    print(f"      🔍 DEBUG: Exception details: {type(e).__name__}: {str(e)[:200]}")
                    
                    # Create error result
                    error_result = ClassificationResult(
                        speech_id="unknown",
                        is_brexit_related=False,
                        confidence_score=0.0,
                        reasoning=f"Future processing error: {str(e)[:200]}",
                        key_indicators=[],
                        provider="unknown",
                        model="unknown",
                        processing_time=0.0,
                        error=str(e)[:500],
                        retry_count=0
                    )
                    batch_results.append(error_result)
                    completed_count += 1

            # Handle any remaining futures
            remaining_futures = [f for f in all_futures if not f.done()]
            if remaining_futures:
                logger.warning(f"⚠️ {len(remaining_futures)} futures noch ausstehend nach {max_wait_time}s")
                print(f"  ⚠️  {len(remaining_futures)} unfertige Requests werden abgebrochen")
                for future in remaining_futures:
                    future.cancel()
                logger.warning(f"⚠️ {len(remaining_futures)} futures wurden abgebrochen")

            print(f"  ✅ {completed_count} von {len(speech_data_list)} Futures verarbeitet")
            print(f"  🔍 DEBUG: {len(batch_results)} Ergebnisse in batch_results gesammelt")

            # Now save all results to database
            print(f"  💾 Speichere {len(batch_results)} Ergebnisse in Datenbank...")
            print(f"  🔍 DEBUG: Starte Datenbank-Speicherung für Batch {batch_num}")
            
            saved_count = 0
            for result in batch_results:
                try:
                    indicators_str = ", ".join(result.key_indicators) if result.key_indicators else ""

                    print(f"    🔍 DEBUG: Speichere speech_id={result.speech_id}, provider={result.provider}, brexit={result.is_brexit_related}")
                    
                    conn.execute("""
                                 UPDATE speeches
                                 SET llm_classified_brexit = ?,
                                     llm_confidence_score  = ?,
                                     llm_reasoning         = ?,
                                     llm_key_indicators    = ?,
                                     llm_provider          = ?,
                                     llm_model             = ?,
                                     llm_processing_time   = ?,
                                     llm_input_tokens      = ?,
                                     llm_output_tokens     = ?,
                                     llm_cost_usd          = ?,
                                     llm_processed         = TRUE,
                                     llm_error             = ?,
                                     llm_retry_count       = ?,
                                     llm_processed_at      = ?
                                 WHERE speech_id = ?
                                 """, [
                                     result.is_brexit_related,
                                     result.confidence_score,
                                     result.reasoning,
                                     indicators_str,
                                     result.provider,
                                     result.model,
                                     result.processing_time,
                                     result.input_tokens,
                                     result.output_tokens,
                                     result.cost_usd,
                                     result.error,
                                     result.retry_count,
                                     datetime.now(),
                                     result.speech_id
                                 ])

                    saved_count += 1
                    print(f"    ✅ DEBUG: Speech {result.speech_id} gespeichert ({saved_count}/{len(batch_results)})")

                    # Update statistics
                    total_processed += 1
                    if result.is_brexit_related:
                        brexit_count += 1

                except Exception as e:
                    logger.error(f"Database save error: {e}")
                    print(f"      ❌ Datenbankfehler beim Speichern: {e}")
                    print(f"      🔍 DEBUG: Fehler beim Speichern von speech_id={result.speech_id}: {e}")

            print(f"  ✅ {saved_count} von {len(batch_results)} Ergebnisse gespeichert")
            
            # WICHTIG: Commit nach jedem Batch (wichtig für Datensicherheit!)
            print(f"  💾 DEBUG: Committe Batch {batch_num} zur Datenbank...")
            conn.commit()
            print(f"  ✅ DEBUG: Batch {batch_num} erfolgreich committed")
            logger.info(f"  ✓ Batch {batch_num} gespeichert und committed")
            
            # Verifiziere Speicherung
            print(f"  🔍 DEBUG: Verifiziere Speicherung...")
            saved_in_db = conn.execute(f"""
                SELECT COUNT(*) FROM speeches 
                WHERE llm_processed = TRUE 
                AND llm_processed_at > (CURRENT_TIMESTAMP - INTERVAL '5 minutes')
            """).fetchone()[0]
            print(f"  📊 DEBUG: {saved_in_db} Speeches in den letzten 5 Minuten gespeichert")
            
            # Zeige Batch-Statistiken
            batch_brexit = conn.execute("""
                SELECT COUNT(*) FROM speeches 
                WHERE llm_classified_brexit = TRUE 
                AND llm_processed_at > (CURRENT_TIMESTAMP - INTERVAL '1 hour')
            """).fetchone()[0]
            
            print(f"  📊 BATCH {batch_num} ABGESCHLOSSEN:")
            print(f"    Verarbeitet: {len(batch_speeches)}")
            print(f"    Brexit-Speeches in diesem Batch: {batch_brexit}")
            print(f"    Gesamt verarbeitet: {total_processed:,}")
            print(f"    Gesamt Brexit-Speeches: {brexit_count:,}")
            print(f"    Fehler: {errors_count}")
            print(f"    Gesamtkosten: ${cost_tracker.total_cost:.2f}")
            
            # Progress tracking
            remaining_speeches = conn.execute("SELECT COUNT(*) FROM speeches WHERE llm_processed = FALSE").fetchone()[0]
            progress_pct = (total_processed / total_speeches * 100) if total_speeches > 0 else 0

            print(f"\n📈 GESAMTFORTSCHRITT:")
            print(f"    Verarbeitet: {total_processed:,} / {total_speeches:,} ({progress_pct:.1f}%)")
            print(f"    Verbleibend: {remaining_speeches:,}")
            print(f"    Budget: ${cost_tracker.total_cost:.2f} / ${MAX_BUDGET_USD:.2f}")

            logger.info(f"📈 Fortschritt: {total_processed:,}/{total_speeches:,} ({progress_pct:.1f}%) | Verbleibend: {remaining_speeches:,}")
            
            # Zeige Pool-Status
            print(f"  🏊 POOL-STATUS:")
            for provider, pool in classifier.rate_limiter.provider_pools.items():
                health_emoji = "🟢" if pool.health_score > 0.7 else "🟡" if pool.health_score > 0.4 else "🔴"
                print(f"    {health_emoji} {provider}: {pool.active_requests}/{pool.current_workers} Worker | Health: {pool.health_score:.2f}")
            
            # Prüfe Budget
            if cost_tracker.budget_exceeded:
                logger.warning("🛑 Budget-Limit erreicht! Beende Verarbeitung...")
                print(f"  🛑 Budget-Limit erreicht! Beende Verarbeitung...")
                break

    except Exception as e:
        logger.error(f"❌ Unerwarteter Fehler im Batch {batch_num}: {e}")
        print(f"  ❌ Unerwarteter Fehler im Batch {batch_num}: {e}")
        # Cancel remaining futures
        for future in all_futures:
            if not future.done():
                future.cancel()

    except KeyboardInterrupt:
        print(f"\n⚠️ Verarbeitung durch Benutzer unterbrochen!")
        logger.warning("\n⚠️ Verarbeitung durch Benutzer unterbrochen!")
    except Exception as e:
        print(f"\n❌ Kritischer Fehler: {e}")
        logger.error(f"\n❌ Kritischer Fehler: {e}")
    finally:
        # Cleanup Provider-Pools
        print(f"\n🧹 Schließe Provider-Pools...")
        for provider, pool in classifier.rate_limiter.provider_pools.items():
            if pool.executor:
                logger.info(f"🧹 Shutting down {provider} pool...")
                pool.executor.shutdown(wait=False, cancel_futures=True)  # Don't wait!
                print(f"  ✓ {provider} Pool geschlossen (forced)")
        
        # Finale Speicherung
        conn.commit()
        
        # Finale Statistiken
        elapsed = time.time() - start_time
        new_processed = total_processed - processed_count
        
        print(f"\n" + "=" * 80)
        print(f"📊 VERARBEITUNGSSTATISTIKEN")
        print(f"=" * 80)
        print(f"✓ Neue verarbeitete Speeches: {new_processed:,}")
        print(f"✓ Gesamt verarbeitet: {total_processed:,} / {total_speeches:,}")
        print(f"✓ Brexit-Speeches: {brexit_count:,}")
        print(f"✓ Fehler: {errors_count:,}")
        print(f"✓ Verarbeitungszeit: {elapsed/60:.1f} Minuten")
        print(f"✓ Durchschnittliche Rate: {new_processed/elapsed:.2f} speeches/s")
        
        logger.info("\n" + "=" * 80)
        logger.info("VERARBEITUNGSSTATISTIKEN")
        logger.info("=" * 80)
        logger.info(f"✓ Neue verarbeitete Speeches: {new_processed:,}")
        logger.info(f"✓ Gesamt verarbeitet: {total_processed:,} / {total_speeches:,}")
        logger.info(f"✓ Brexit-Speeches: {brexit_count:,}")
        logger.info(f"✓ Fehler: {errors_count:,}")
        logger.info(f"✓ Verarbeitungszeit: {elapsed/60:.1f} Minuten")
        logger.info(f"✓ Durchschnittliche Rate: {new_processed/elapsed:.2f} speeches/s")
        
        # Pool-Status-Zusammenfassung
        print(f"\n" + "=" * 80)
        print(f"🏊 POOL-STATUS ZUSAMMENFASSUNG")
        print(f"=" * 80)
        for provider, pool in classifier.rate_limiter.provider_pools.items():
            health_emoji = "🟢" if pool.health_score > 0.7 else "🟡" if pool.health_score > 0.4 else "🔴"
            print(f"{health_emoji} {provider}: {pool.current_workers} Worker | Health: {pool.health_score:.2f} | "
                  f"Successes: {pool.consecutive_successes} | Errors: {pool.consecutive_errors}")
        
        # Kosten-Zusammenfassung
        cost_summary = cost_tracker.get_summary()
        print(f"\n" + "=" * 80)
        print(f"💰 KOSTENÜBERSICHT")
        print(f"=" * 80)
        print(f"💰 Gesamtkosten: ${cost_summary['total_cost']:.2f}")
        print(f"💰 Budget verwendet: {cost_summary['budget_used_percent']:.1f}%")
        print(f"💰 Verbleibendes Budget: ${cost_summary['remaining_budget']:.2f}")
        
        logger.info("\n" + "=" * 80)
        logger.info("KOSTENÜBERSICHT")
        logger.info("=" * 80)
        logger.info(f"💰 Gesamtkosten: ${cost_summary['total_cost']:.2f}")
        logger.info(f"💰 Budget verwendet: {cost_summary['budget_used_percent']:.1f}%")
        logger.info(f"💰 Verbleibendes Budget: ${cost_summary['remaining_budget']:.2f}")
        
        if cost_summary['cost_by_provider']:
            print(f"\nKosten nach Provider:")
            for provider, cost in sorted(cost_summary['cost_by_provider'].items(), key=lambda x: x[1], reverse=True):
                print(f"  • {provider}: ${cost:.2f}")
            
            logger.info("\nKosten nach Provider:")
            for provider, cost in sorted(cost_summary['cost_by_provider'].items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  • {provider}: ${cost:.2f}")
        
        if cost_summary['cost_by_model']:
            print(f"\nKosten nach Modell:")
            for model, cost in sorted(cost_summary['cost_by_model'].items(), key=lambda x: x[1], reverse=True):
                print(f"  • {model}: ${cost:.2f}")
            
            logger.info("\nKosten nach Modell:")
            for model, cost in sorted(cost_summary['cost_by_model'].items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  • {model}: ${cost:.2f}")

def show_final_statistics(conn):
    """Zeigt finale Statistiken"""
    print("\n" + "=" * 80)
    print("FINALE DATENBANK-STATISTIKEN")
    print("=" * 80)
    
    # Grundlegende Statistiken
    total_speeches = conn.execute("SELECT COUNT(*) FROM speeches").fetchone()[0]
    processed_speeches = conn.execute("SELECT COUNT(*) FROM speeches WHERE llm_processed = TRUE").fetchone()[0]
    brexit_speeches = conn.execute("SELECT COUNT(*) FROM speeches WHERE llm_classified_brexit = TRUE").fetchone()[0]
    error_count = conn.execute("SELECT COUNT(*) FROM speeches WHERE llm_error IS NOT NULL").fetchone()[0]
    
    print(f"\nGrundlegende Statistiken:")
    print(f"  Gesamt Speeches:           {total_speeches:,}")
    print(f"  Verarbeitete Speeches:     {processed_speeches:,}")
    print(f"  Brexit-Speeches:           {brexit_speeches:,}")
    print(f"  Brexit-Rate:               {brexit_speeches/max(processed_speeches,1)*100:.1f}%")
    print(f"  Fehler:                    {error_count:,}")
    
    # Provider-Statistiken
    provider_stats = conn.execute("""
        SELECT llm_provider, 
               COUNT(*) as count,
               AVG(llm_confidence_score) as avg_confidence,
               AVG(llm_processing_time) as avg_time,
               SUM(llm_cost_usd) as total_cost
        FROM speeches 
        WHERE llm_processed = TRUE
        GROUP BY llm_provider
        ORDER BY count DESC
    """).fetchall()
    
    print(f"\nProvider-Statistiken:")
    for provider, count, avg_conf, avg_time, total_cost in provider_stats:
        if provider:
            print(f"  {provider}: {count:,} speeches | "
                  f"Conf: {avg_conf:.3f} | "
                  f"Zeit: {avg_time:.2f}s | "
                  f"Kosten: ${total_cost:.2f}")
    
    # Model-Statistiken
    model_stats = conn.execute("""
        SELECT llm_model, 
               COUNT(*) as count,
               SUM(llm_cost_usd) as total_cost,
               AVG(llm_cost_usd) as avg_cost
        FROM speeches 
        WHERE llm_processed = TRUE
        GROUP BY llm_model
        ORDER BY count DESC
    """).fetchall()
    
    print(f"\nModel-Verteilung:")
    for model, count, total_cost, avg_cost in model_stats:
        if model:
            print(f"  {model}: {count:,} speeches | "
                  f"Total: ${total_cost:.2f} | "
                  f"Avg: ${avg_cost*1000:.4f}/1k")
    
    # Confidence Score Verteilung
    confidence_dist = conn.execute("""
        SELECT 
            CASE 
                WHEN llm_confidence_score >= 0.9 THEN '0.9-1.0 (Sehr hoch)'
                WHEN llm_confidence_score >= 0.7 THEN '0.7-0.9 (Hoch)'
                WHEN llm_confidence_score >= 0.5 THEN '0.5-0.7 (Mittel)'
                WHEN llm_confidence_score >= 0.3 THEN '0.3-0.5 (Niedrig)'
                ELSE '0.0-0.3 (Sehr niedrig)'
            END as confidence_range,
            COUNT(*) as count
        FROM speeches
        WHERE llm_processed = TRUE
        GROUP BY confidence_range
        ORDER BY confidence_range DESC
    """).fetchall()
    
    print(f"\nConfidence Score Verteilung:")
    for range_name, count in confidence_dist:
        print(f"  {range_name}: {count:,} speeches")
    
    # Top Brexit-Indikatoren
    print(f"\nTop Brexit-Indikatoren:")
    indicator_stats = conn.execute("""
        SELECT llm_key_indicators, COUNT(*) as count
        FROM speeches 
        WHERE llm_classified_brexit = TRUE 
          AND llm_key_indicators IS NOT NULL 
          AND llm_key_indicators != ''
        GROUP BY llm_key_indicators
        ORDER BY count DESC
        LIMIT 10
    """).fetchall()
    
    for indicators, count in indicator_stats:
        if indicators:
            print(f"  '{indicators[:60]}...': {count:,} speeches")
    
    # Kosten-Zusammenfassung aus DB
    total_cost = conn.execute("SELECT SUM(llm_cost_usd) FROM speeches WHERE llm_processed = TRUE").fetchone()[0]
    if total_cost:
        print(f"\n💰 Gesamtkosten (aus DB): ${total_cost:.2f}")

def main():
    """Hauptfunktion"""
    print("\n" + "🚀" * 40)
    print("STARTE BREXIT SPEECH KLASSIFIKATION")
    print("🚀" * 40)
    
    print(f"\n📊 KONFIGURATION:")
    print(f"  Input-DB: {INPUT_DB}")
    print(f"  Output-DB: {OUTPUT_DB}")
    print(f"  Budget-Limit: ${MAX_BUDGET_USD}")
    print(f"  Batch-Größe: {BATCH_SIZE}")
    print(f"  Provider-Pools: Dynamisch basierend auf RPM-Limits")
    
    # Test-Modus prüfen
    test_mode = os.getenv('TEST_MODE', 'false').lower() == 'true'
    
    # Verwende immer die bestehende Datenbank (keine Kopie)
    print(f"🔗 Verbinde mit bestehender Datenbank...")
    logger.info(f"Verbinde mit bestehender Datenbank: {INPUT_DB}")
    
    conn = duckdb.connect(INPUT_DB)
    
    # Prüfe und füge notwendige Spalten hinzu
    print(f"🔍 Prüfe Datenbank-Schema...")
    try:
        # Prüfe ob LLM-Spalten existieren
        conn.execute("SELECT llm_processed FROM speeches LIMIT 1")
        print(f"✅ LLM-Spalten bereits vorhanden")
    except:
        print(f"🔧 Füge LLM-Spalten hinzu...")
        conn.execute("""
            ALTER TABLE speeches ADD COLUMN IF NOT EXISTS llm_processed BOOLEAN DEFAULT FALSE;
            ALTER TABLE speeches ADD COLUMN IF NOT EXISTS llm_classified_brexit BOOLEAN DEFAULT FALSE;
            ALTER TABLE speeches ADD COLUMN IF NOT EXISTS llm_confidence_score FLOAT;
            ALTER TABLE speeches ADD COLUMN IF NOT EXISTS llm_reasoning TEXT;
            ALTER TABLE speeches ADD COLUMN IF NOT EXISTS llm_key_indicators VARCHAR;
            ALTER TABLE speeches ADD COLUMN IF NOT EXISTS llm_provider VARCHAR;
            ALTER TABLE speeches ADD COLUMN IF NOT EXISTS llm_model VARCHAR;
            ALTER TABLE speeches ADD COLUMN IF NOT EXISTS llm_cost_usd FLOAT;
            ALTER TABLE speeches ADD COLUMN IF NOT EXISTS llm_processing_time FLOAT;
            ALTER TABLE speeches ADD COLUMN IF NOT EXISTS llm_input_tokens INTEGER;
            ALTER TABLE speeches ADD COLUMN IF NOT EXISTS llm_output_tokens INTEGER;
            ALTER TABLE speeches ADD COLUMN IF NOT EXISTS llm_error TEXT;
            ALTER TABLE speeches ADD COLUMN IF NOT EXISTS llm_processed_at TIMESTAMP;
        """)
        print(f"✅ LLM-Spalten hinzugefügt")
    
    print(f"✅ Datenbank verbunden")
    
    try:
        # Verarbeite Speeches mit Budget-Kontrolle
        print(f"\n🚀 Starte Klassifikationsprozess...")
        process_speeches_with_budget(conn)
        
        # Zeige finale Statistiken
        print(f"\n📈 Zeige finale Statistiken...")
        show_final_statistics(conn)
        
        print(f"\n✅ Klassifikation abgeschlossen!")
        print(f"📁 Output-Datenbank: {OUTPUT_DB}")
        print(f"📊 Log-Datei: llm_classification.log")
        
    except Exception as e:
        logger.error(f"❌ Kritischer Fehler in main(): {e}")
        print(f"❌ FEHLER: {e}")
    finally:
        conn.close()
        print("\n" + "=" * 80)
        print("PROGRAMM BEENDET")
        print("=" * 80)

if __name__ == "__main__":
    main()