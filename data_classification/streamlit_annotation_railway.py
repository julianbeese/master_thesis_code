#!/usr/bin/env python3
"""
Streamlit Annotation Interface für Railway Deployment
Optimiert für Railway PostgreSQL und Multi-User-Zugriff
"""

import streamlit as st
import psycopg2
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
import os
from psycopg2.extras import RealDictCursor
from psycopg2 import pool
import threading
import time
import functools

# Railway PostgreSQL Konfiguration (Fallback für lokale Entwicklung)
DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'frame_classification'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '')
}

# Connection Pool für bessere Performance
_connection_pool = None
_pool_lock = threading.Lock()

class DatabasePool:
    """Thread-safe database connection pool"""
    
    def __init__(self):
        self.pool = None
        self.lock = threading.Lock()
    
    def get_pool(self):
        """Get or create connection pool"""
        if self.pool is None:
            with self.lock:
                if self.pool is None:
                    try:
                        database_url = os.getenv('DATABASE_URL')
                        if database_url:
                            # Parse DATABASE_URL for connection pool
                            self.pool = psycopg2.pool.ThreadedConnectionPool(
                                minconn=1,
                                maxconn=10,
                                dsn=database_url
                            )
                        else:
                            # Use config for local development
                            self.pool = psycopg2.pool.ThreadedConnectionPool(
                                minconn=1,
                                maxconn=5,
                                **DATABASE_CONFIG
                            )
                    except Exception as e:
                        st.error(f"Fehler beim Erstellen des Connection Pools: {e}")
                        return None
        return self.pool
    
    def get_connection(self):
        """Get connection from pool"""
        pool = self.get_pool()
        if pool:
            try:
                return pool.getconn()
            except Exception as e:
                st.error(f"Fehler beim Abrufen der Verbindung: {e}")
                return None
        return None
    
    def return_connection(self, conn):
        """Return connection to pool"""
        pool = self.get_pool()
        if pool and conn:
            try:
                pool.putconn(conn)
            except Exception as e:
                st.error(f"Fehler beim Zurückgeben der Verbindung: {e}")
    
    def close_all(self):
        """Close all connections in pool"""
        if self.pool:
            try:
                self.pool.closeall()
            except Exception as e:
                st.error(f"Fehler beim Schließen des Pools: {e}")

# Global pool instance
_db_pool = DatabasePool()

# Performance Monitoring (reduziert für Railway Rate Limits)
def log_performance(func_name: str, start_time: float):
    """Loggt Performance-Metriken (nur bei sehr langsamen Operationen)"""
    duration = time.time() - start_time
    if duration > 5.0:  # Nur bei sehr langsamen Operationen loggen (>5s)
        st.warning(f"⚠️ {func_name} dauerte {duration:.2f}s - Performance-Optimierung empfohlen")

# Caching für bessere Performance (längere TTL für Railway)
@st.cache_data(ttl=600)  # Cache für 10 Minuten
def cached_get_statistics() -> Dict[str, Any]:
    """Gecachte Version der Statistiken"""
    start_time = time.time()
    result = get_statistics()
    log_performance("get_statistics", start_time)
    return result

@st.cache_data(ttl=300)  # Cache für 5 Minuten
def cached_get_classified_chunks() -> List[Dict[str, Any]]:
    """Gecachte Version der klassifizierten Chunks"""
    start_time = time.time()
    result = get_classified_chunks()
    log_performance("get_classified_chunks", start_time)
    return result

@st.cache_data(ttl=300)  # Cache für 5 Minuten
def cached_get_annotation_conflicts() -> List[Dict[str, Any]]:
    """Gecachte Version der Annotation-Konflikte"""
    start_time = time.time()
    result = get_annotation_conflicts()
    log_performance("get_annotation_conflicts", start_time)
    return result

# Frame-Kategorien
FRAME_CATEGORIES = [
    "Human Impact",
    "Powerlessness", 
    "Economic",
    "Moral Value",
    "Conflict",
    "None/Not Relevant",
    "t.b.d."
]

# Brexit-Position Kategorien
BREXIT_POSITION_CATEGORIES = [
    "Pro-Brexit",
    "Anti-Brexit", 
    "Neutral/Unclear",
    "Not Applicable"
]

def get_db_connection():
    """Erstellt PostgreSQL Verbindung für Railway mit Connection Pool"""
    try:
        conn = _db_pool.get_connection()
        if conn is None:
            st.error("❌ Keine Datenbankverbindung verfügbar!")
            st.error("🔧 Bitte überprüfe die DATABASE_URL Konfiguration")
            return None
        return conn
    except Exception as e:
        st.error(f"❌ Fehler bei Datenbankverbindung: {e}")
        st.error(f"🔧 DATABASE_URL: {os.getenv('DATABASE_URL', 'Nicht gesetzt')}")
        return None

def return_db_connection(conn):
    """Gibt Verbindung an Pool zurück"""
    if conn:
        _db_pool.return_connection(conn)

def cleanup_connections():
    """Bereinigt alle Verbindungen beim Beenden"""
    try:
        _db_pool.close_all()
    except Exception as e:
        st.error(f"Fehler beim Schließen der Verbindungen: {e}")

def safe_rerun():
    """Rate-limited rerun um WebSocket-Fehler zu reduzieren"""
    current_time = time.time()
    if current_time - st.session_state.get('last_rerun', 0) >= 0.5:  # Mindestens 0.5 Sekunden zwischen Reruns
        st.session_state.last_rerun = current_time
        st.rerun()

def init_session_state():
    """Initialisiert Session State"""
    if 'chunks' not in st.session_state:
        st.session_state.chunks = []
    if 'current_chunk_index' not in st.session_state:
        st.session_state.current_chunk_index = 0
    if 'annotations' not in st.session_state:
        st.session_state.annotations = {}
    if 'user_name' not in st.session_state:
        st.session_state.user_name = ""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None

def create_tables_if_not_exist():
    """Erstellt Tabellen falls sie nicht existieren"""
    conn = get_db_connection()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        
        # Chunks-Tabelle
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id VARCHAR(255) PRIMARY KEY,
                speech_id VARCHAR(255),
                debate_id VARCHAR(255),
                speaker_name VARCHAR(255),
                speaker_party VARCHAR(255),
                debate_title TEXT,
                debate_date DATE,
                chunk_text TEXT,
                chunk_index INTEGER,
                total_chunks INTEGER,
                word_count INTEGER,
                char_count INTEGER,
                chunking_method VARCHAR(100),
                assigned_user VARCHAR(255),
                frame_label VARCHAR(100),
                annotation_confidence INTEGER,
                annotation_notes TEXT,
                pre_brexit BOOLEAN,
                brexit_position VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Agreement-Tabelle
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agreement_chunks (
                chunk_id VARCHAR(255) PRIMARY KEY,
                annotator1 VARCHAR(255),
                annotator2 VARCHAR(255),
                label1 VARCHAR(100),
                label2 VARCHAR(100),
                agreement_score DECIMAL(3,2),
                agreement_perfect BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Annotation-History-Tabelle für echte Konflikte
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS annotation_history (
                id SERIAL PRIMARY KEY,
                chunk_id VARCHAR(255),
                user_name VARCHAR(255),
                frame_label VARCHAR(100),
                brexit_position VARCHAR(100),
                annotation_notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
            );
        """)
        
        conn.commit()
        
        # Indizes für bessere Performance (separate Transaktionen)
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_assigned_user ON chunks(assigned_user);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_frame_label ON chunks(frame_label);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_brexit_position ON chunks(brexit_position);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_pre_brexit ON chunks(pre_brexit);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_speaker_name ON chunks(speaker_name);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_debate_date ON chunks(debate_date);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_updated_at ON chunks(updated_at);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_annotation_history_chunk_id ON annotation_history(chunk_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_annotation_history_user_name ON annotation_history(user_name);")
            conn.commit()
        except Exception as e:
            st.warning(f"⚠️ Einige Indizes konnten nicht erstellt werden: {e}")
            conn.rollback()
        
        # Füge neue Spalten hinzu falls sie nicht existieren (separate Transaktionen)
        try:
            cursor.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS pre_brexit BOOLEAN;")
            conn.commit()
        except Exception as e:
            st.warning(f"⚠️ Spalte pre_brexit existiert bereits oder konnte nicht hinzugefügt werden: {e}")
            conn.rollback()
        
        try:
            cursor.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS brexit_position VARCHAR(100);")
            conn.commit()
        except Exception as e:
            st.warning(f"⚠️ Spalte brexit_position existiert bereits oder konnte nicht hinzugefügt werden: {e}")
            conn.rollback()
        
        # Aktualisiere pre_brexit Spalte basierend auf debate_date
        try:
            cursor.execute("""
                UPDATE chunks 
                SET pre_brexit = (debate_date < '2016-06-23'::date)
                WHERE pre_brexit IS NULL
            """)
            conn.commit()
        except Exception as e:
            st.warning(f"⚠️ Pre-Brexit Update fehlgeschlagen: {e}")
            conn.rollback()
        
        cursor.close()
        # Erfolgsmeldung nur bei Debug-Modus
        if os.getenv('DEBUG', 'false').lower() == 'true':
            st.success("✅ Tabellen erfolgreich erstellt/aktualisiert!")
        
    except Exception as e:
        st.error(f"❌ Fehler beim Erstellen der Tabellen: {e}")
        if conn:
            conn.rollback()
    finally:
        return_db_connection(conn)

def load_database_chunks(user_name: str = None, limit: int = None, only_unannotated: bool = True) -> List[Dict[str, Any]]:
    """Lädt Chunks aus PostgreSQL für einen bestimmten User"""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        if user_name:
            if only_unannotated:
                # Lade nur unklassifizierte Chunks für den spezifischen User
                if limit:
                    query = """
                    SELECT * FROM chunks 
                    WHERE assigned_user = %s 
                    AND (frame_label IS NULL OR frame_label = '')
                    ORDER BY chunk_id 
                    LIMIT %s
                    """
                    cursor.execute(query, (user_name, limit))
                else:
                    query = """
                    SELECT * FROM chunks 
                    WHERE assigned_user = %s 
                    AND (frame_label IS NULL OR frame_label = '')
                    ORDER BY chunk_id
                    """
                    cursor.execute(query, (user_name,))
            else:
                # Lade alle Chunks für den spezifischen User (inkl. klassifizierte)
                if limit:
                    query = """
                    SELECT * FROM chunks 
                    WHERE assigned_user = %s
                    ORDER BY chunk_id 
                    LIMIT %s
                    """
                    cursor.execute(query, (user_name, limit))
                else:
                    query = """
                    SELECT * FROM chunks 
                    WHERE assigned_user = %s
                    ORDER BY chunk_id
                    """
                    cursor.execute(query, (user_name,))
        else:
            # Lade alle unzugewiesenen Chunks (für Admin-View)
            if limit:
                query = """
                SELECT * FROM chunks 
                WHERE assigned_user IS NULL OR assigned_user = ''
                ORDER BY chunk_id 
                LIMIT %s
                """
                cursor.execute(query, (limit,))
            else:
                query = """
                SELECT * FROM chunks 
                WHERE assigned_user IS NULL OR assigned_user = ''
                ORDER BY chunk_id
                """
                cursor.execute(query)
        
        chunks = cursor.fetchall()
        cursor.close()
        
        # Konvertiere zu Dictionary-Liste
        chunk_list = []
        for chunk in chunks:
            chunk_dict = dict(chunk)
            chunk_list.append(chunk_dict)
        
        return chunk_list
        
    except Exception as e:
        st.error(f"❌ Fehler beim Laden der Chunks: {e}")
        st.error("🔧 Bitte versuche es erneut oder kontaktiere den Administrator")
        if conn:
            conn.rollback()
        return []
    finally:
        return_db_connection(conn)

def update_database_annotation(chunk_id: str, frame_label: str, confidence: int, notes: str, user_name: str, brexit_position: str = None):
    """Aktualisiert Annotation in PostgreSQL"""
    conn = get_db_connection()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        
        # Aktualisiere chunks-Tabelle
        update_sql = """
        UPDATE chunks 
        SET frame_label = %s, annotation_confidence = %s, annotation_notes = %s, 
            assigned_user = %s, brexit_position = %s, updated_at = CURRENT_TIMESTAMP
        WHERE chunk_id = %s
        """
        
        cursor.execute(update_sql, (frame_label, confidence, notes, user_name, brexit_position, chunk_id))
        
        # Füge auch zur Annotation-History hinzu
        history_sql = """
        INSERT INTO annotation_history (chunk_id, user_name, frame_label, brexit_position, annotation_notes)
        VALUES (%s, %s, %s, %s, %s)
        """
        
        cursor.execute(history_sql, (chunk_id, user_name, frame_label, brexit_position, notes))
        
        conn.commit()
        cursor.close()
        
    except Exception as e:
        st.error(f"Fehler beim Aktualisieren der Datenbank: {e}")
        if conn:
            conn.rollback()
    finally:
        return_db_connection(conn)

def get_statistics() -> Dict[str, Any]:
    """Berechnet Statistiken"""
    conn = get_db_connection()
    if not conn:
        return {}
    
    try:
        cursor = conn.cursor()
        
        # Gesamt-Statistiken
        cursor.execute("""
            SELECT 
                COUNT(*) as total_chunks,
                COUNT(CASE WHEN frame_label IS NOT NULL THEN 1 END) as annotated_chunks,
                COUNT(CASE WHEN assigned_user IS NOT NULL AND assigned_user != '' THEN 1 END) as assigned_chunks
            FROM chunks
        """)
        total_stats = cursor.fetchone()
        
        # Frame-Verteilung
        cursor.execute("""
            SELECT frame_label, COUNT(*) as count
            FROM chunks 
            WHERE frame_label IS NOT NULL
            GROUP BY frame_label
            ORDER BY count DESC
        """)
        frame_stats = cursor.fetchall()
        
        # User-Statistiken
        cursor.execute("""
            SELECT assigned_user, COUNT(*) as count
            FROM chunks 
            WHERE assigned_user IS NOT NULL AND assigned_user != ''
            GROUP BY assigned_user
            ORDER BY count DESC
        """)
        user_stats = cursor.fetchall()
        
        # Brexit-Position Statistiken
        cursor.execute("""
            SELECT brexit_position, COUNT(*) as count
            FROM chunks 
            WHERE brexit_position IS NOT NULL AND brexit_position != ''
            GROUP BY brexit_position
            ORDER BY count DESC
        """)
        brexit_stats = cursor.fetchall()
        
        # Pre-Brexit vs Post-Brexit Statistiken
        cursor.execute("""
            SELECT 
                COUNT(CASE WHEN pre_brexit = true THEN 1 END) as pre_brexit_count,
                COUNT(CASE WHEN pre_brexit = false THEN 1 END) as post_brexit_count
            FROM chunks
        """)
        brexit_timing_stats = cursor.fetchone()
        
        cursor.close()
        
        return {
            'total_chunks': total_stats[0],
            'annotated_chunks': total_stats[1],
            'assigned_chunks': total_stats[2],
            'by_frame': {frame: count for frame, count in frame_stats},
            'by_user': {user: count for user, count in user_stats},
            'by_brexit_position': {position: count for position, count in brexit_stats},
            'pre_brexit_count': brexit_timing_stats[0],
            'post_brexit_count': brexit_timing_stats[1]
        }
        
    except Exception as e:
        st.error(f"Fehler beim Laden der Statistiken: {e}")
        if conn:
            conn.rollback()
        return {}
    finally:
        return_db_connection(conn)

def show_statistics():
    """Zeigt Statistiken"""
    stats = cached_get_statistics()
    
    if not stats:
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Gesamt Chunks", f"{stats['total_chunks']:,}")
    
    with col2:
        st.metric("Annotiert", f"{stats['annotated_chunks']:,}")
    
    with col3:
        st.metric("Zugewiesen", f"{stats['assigned_chunks']:,}")
    
    # Frame-Verteilung
    if stats['by_frame']:
        st.subheader("📊 Frame-Verteilung")
        
        try:
            frame_data = []
            for frame, count in stats['by_frame'].items():
                frame_data.append({'Frame': str(frame), 'Anzahl': int(count)})
            
            if frame_data:
                frame_df = pd.DataFrame(frame_data)
                st.bar_chart(frame_df.set_index('Frame'))
            else:
                st.info("Keine Frame-Daten verfügbar")
        except Exception as e:
            st.warning(f"Konnte Chart nicht anzeigen: {e}")
            st.write("**Frame-Verteilung:**")
            for frame, count in stats['by_frame'].items():
                st.write(f"- {frame}: {count}")
    
    # User-Verteilung
    if stats['by_user']:
        st.subheader("👥 User-Verteilung")
        for user, count in stats['by_user'].items():
            st.write(f"- **{user}**: {count:,} Chunks")
    
    # Brexit-Timing Verteilung
    if 'pre_brexit_count' in stats and 'post_brexit_count' in stats:
        st.subheader("📅 Brexit-Timing")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Pre-Brexit Chunks", f"{stats['pre_brexit_count']:,}")
        with col2:
            st.metric("Post-Brexit Chunks", f"{stats['post_brexit_count']:,}")
    
    # Brexit-Position Verteilung
    if stats.get('by_brexit_position'):
        st.subheader("🇬🇧 Brexit-Positionen")
        
        try:
            brexit_data = []
            for position, count in stats['by_brexit_position'].items():
                brexit_data.append({'Position': str(position), 'Anzahl': int(count)})
            
            if brexit_data:
                brexit_df = pd.DataFrame(brexit_data)
                st.bar_chart(brexit_df.set_index('Position'))
            else:
                st.info("Keine Brexit-Position Daten verfügbar")
        except Exception as e:
            st.warning(f"Konnte Chart nicht anzeigen: {e}")
            st.write("**Brexit-Positionen:**")
            for position, count in stats['by_brexit_position'].items():
                st.write(f"- {position}: {count}")

def show_chunk_annotation():
    """Zeigt Chunk-Annotation Interface"""
    if not st.session_state.chunks:
        st.warning("Keine Chunks geladen!")
        return
    
    current_chunk = st.session_state.chunks[st.session_state.current_chunk_index]
    chunk_id = current_chunk['chunk_id']
    
    # Chunk-Informationen
    st.subheader(f"📝 Chunk {st.session_state.current_chunk_index + 1} von {len(st.session_state.chunks)}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**ID:** {chunk_id}")
    with col2:
        st.write(f"**Wörter:** {current_chunk['word_count']}")
    with col3:
        st.write(f"**Zeichen:** {current_chunk['char_count']}")
    
    # Speaker-Informationen
    st.write(f"**Speaker:** {current_chunk['speaker_name']} ({current_chunk['speaker_party']})")
    st.write(f"**Debatte:** {current_chunk['debate_title']}")
    st.write(f"**Datum:** {current_chunk['debate_date']}")
    
    # Pre-Brexit Status
    pre_brexit = current_chunk.get('pre_brexit', False)
    if pre_brexit:
        st.info("🇬🇧 **Pre-Brexit Chunk** - Dieser Chunk stammt aus der Zeit vor dem Brexit-Referendum (23. Juni 2016)")
    else:
        st.info("📅 **Post-Brexit Chunk** - Dieser Chunk stammt aus der Zeit nach dem Brexit-Referendum")
    
    # Chunk-Text
    st.subheader("📄 Chunk-Text")
    st.text_area("Chunk-Text", current_chunk['chunk_text'], height=200, disabled=True, label_visibility="collapsed")
    
    # Annotation-Formular
    st.subheader("🏷️ Frame-Annotation")
    
    # Frame-Auswahl
    frame_label = st.selectbox(
        "Frame-Kategorie:",
        options=[""] + FRAME_CATEGORIES,
        key=f"frame_{chunk_id}"
    )
    
    # Brexit-Position (nur für Pre-Brexit Chunks)
    brexit_position = None
    if pre_brexit:
        st.subheader("🇬🇧 Brexit-Position")
        brexit_position = st.selectbox(
            "Position des Sprechers zum Brexit:",
            options=[""] + BREXIT_POSITION_CATEGORIES,
            key=f"brexit_{chunk_id}",
            help="Nur für Pre-Brexit Chunks relevant"
        )
    
    # Notas
    notes = st.text_area(
        "Notizen:",
        placeholder="Optionale Notizen zur Annotation...",
        key=f"notes_{chunk_id}"
    )
    
    # Buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Speichern", type="primary"):
            if frame_label:
                # Prüfe ob Brexit-Position für Pre-Brexit Chunks erforderlich ist
                if pre_brexit and not brexit_position:
                    st.error("Bitte wähle eine Brexit-Position für Pre-Brexit Chunks!")
                    return
                
                # Aktualisiere Datenbank
                update_database_annotation(
                    chunk_id, frame_label, 3, notes, st.session_state.user_name, brexit_position
                )
                
                st.success("✅ Annotation gespeichert!")
                
                # Automatisch zum nächsten Chunk
                if st.session_state.current_chunk_index < len(st.session_state.chunks) - 1:
                    st.session_state.current_chunk_index += 1
                    # Formularfelder zurücksetzen
                    if f"frame_{chunk_id}" in st.session_state:
                        del st.session_state[f"frame_{chunk_id}"]
                    if f"brexit_{chunk_id}" in st.session_state:
                        del st.session_state[f"brexit_{chunk_id}"]
                    if f"notes_{chunk_id}" in st.session_state:
                        del st.session_state[f"notes_{chunk_id}"]
                    st.info("🔄 Lade nächsten Chunk...")
                else:
                    st.info("🎉 Alle Chunks annotiert!")
                
                # Rate-limited rerun
                safe_rerun()
            else:
                st.error("Bitte wähle eine Frame-Kategorie!")
    
    with col2:
        if st.button("⏭️ Nächster"):
            if st.session_state.current_chunk_index < len(st.session_state.chunks) - 1:
                st.session_state.current_chunk_index += 1
                safe_rerun()
            else:
                st.info("Letzter Chunk erreicht!")
    
    with col3:
        if st.button("⏮️ Vorheriger"):
            if st.session_state.current_chunk_index > 0:
                st.session_state.current_chunk_index -= 1
                safe_rerun()
            else:
                st.info("Erster Chunk erreicht!")
    
    # Navigation
    st.subheader("🧭 Navigation")
    
    # Chunk-Index
    new_index = st.number_input(
        "Gehe zu Chunk:",
        min_value=1,
        max_value=len(st.session_state.chunks),
        value=st.session_state.current_chunk_index + 1
    )
    
    if st.button("Gehe zu Chunk"):
        st.session_state.current_chunk_index = new_index - 1
        safe_rerun()

def get_classified_chunks() -> List[Dict[str, Any]]:
    """Lädt alle klassifizierten Chunks aus der Datenbank"""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT 
                chunk_id,
                speech_id,
                speaker_name,
                speaker_party,
                debate_title,
                debate_date,
                chunk_text,
                frame_label,
                brexit_position,
                assigned_user,
                annotation_notes,
                word_count,
                char_count,
                pre_brexit,
                created_at,
                updated_at
            FROM chunks
            WHERE frame_label IS NOT NULL 
            AND frame_label != ''
            ORDER BY updated_at DESC
        """)
        
        chunks = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Konvertiere zu Dictionary-Liste
        chunk_list = []
        for chunk in chunks:
            chunk_dict = dict(chunk)
            chunk_list.append(chunk_dict)
        
        return chunk_list
        
    except Exception as e:
        st.error(f"Fehler beim Laden der klassifizierten Chunks: {e}")
        if conn:
            conn.close()
        return []

def get_annotation_conflicts() -> List[Dict[str, Any]]:
    """Erkennt echte Annotation-Konflikte zwischen verschiedenen Usern (inkl. _dup Chunks)"""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Konflikterkennung für Original vs _dup Chunks (nur Frame-Labels)
        cursor.execute("""
            WITH chunk_pairs AS (
                -- Finde Original-Chunks und ihre _dup Gegenstücke
                SELECT 
                    c1.chunk_id as original_chunk,
                    c2.chunk_id as dup_chunk,
                    c1.assigned_user as user1,
                    c1.frame_label as frame1,
                    c1.brexit_position as brexit1,
                    c1.annotation_notes as notes1,
                    c1.updated_at as created1,
                    c1.speech_id,
                    c1.speaker_name,
                    c1.speaker_party,
                    c1.debate_title,
                    c1.debate_date,
                    c1.chunk_text,
                    c2.assigned_user as user2,
                    c2.frame_label as frame2,
                    c2.brexit_position as brexit2,
                    c2.annotation_notes as notes2,
                    c2.updated_at as created2
                FROM chunks c1
                JOIN chunks c2 ON c1.chunk_id = REPLACE(c2.chunk_id, '_dup', '')
                WHERE c1.chunk_id NOT LIKE '%_dup'
                AND c2.chunk_id LIKE '%_dup'
                AND c1.frame_label IS NOT NULL AND c1.frame_label != ''
                AND c2.frame_label IS NOT NULL AND c2.frame_label != ''
                AND c1.frame_label != c2.frame_label
                -- Filtere nur Konflikte zwischen "t.b.d." und "None/Not Relevant" heraus
                AND NOT (
                    (c1.frame_label = 't.b.d.' AND c2.frame_label = 'None/Not Relevant')
                    OR (c1.frame_label = 'None/Not Relevant' AND c2.frame_label = 't.b.d.')
                )
            )
            SELECT DISTINCT * FROM chunk_pairs
            ORDER BY created2 DESC
        """)
        
        conflicts = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Konvertiere zu Dictionary-Liste
        conflict_list = []
        for conflict in conflicts:
            conflict_dict = dict(conflict)
            conflict_list.append(conflict_dict)
        
        return conflict_list
        
    except Exception as e:
        st.error(f"Fehler beim Laden der Konflikte: {e}")
        if conn:
            conn.close()
        return []

def resolve_conflict(chunk_id: str, final_frame: str, final_brexit: str, final_notes: str, resolved_by: str):
    """Löst einen Annotation-Konflikt durch finale Entscheidung"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Aktualisiere BEIDE Chunks (Original und Duplikat) mit der finalen Entscheidung
        update_sql = """
        UPDATE chunks 
        SET frame_label = %s, brexit_position = %s, annotation_notes = %s, 
            assigned_user = %s, updated_at = CURRENT_TIMESTAMP
        WHERE chunk_id = %s OR chunk_id = %s
        """
        
        # Erstelle Duplikat-Chunk-ID
        dup_chunk_id = chunk_id + '_dup'
        
        cursor.execute(update_sql, (final_frame, final_brexit, final_notes, resolved_by, chunk_id, dup_chunk_id))
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        st.error(f"Fehler beim Lösen des Konflikts: {e}")
        if conn:
            conn.rollback()
            conn.close()
        return False

def get_double_classification_stats() -> Dict[str, Any]:
    """Berechnet Statistiken über doppelt klassifizierte Chunks"""
    conn = get_db_connection()
    if not conn:
        return {}
    
    try:
        cursor = conn.cursor()
        
        # Doppelt klassifizierte Chunks (mehrere User haben denselben Chunk annotiert)
        cursor.execute("""
            WITH user_annotation_counts AS (
                SELECT 
                    chunk_id,
                    COUNT(DISTINCT user_name) as user_count,
                    COUNT(*) as total_annotations
                FROM annotation_history
                WHERE frame_label IS NOT NULL AND frame_label != ''
                GROUP BY chunk_id
                HAVING COUNT(DISTINCT user_name) > 1
            ),
            double_classified_details AS (
                SELECT 
                    c.chunk_id,
                    c.speech_id,
                    c.speaker_name,
                    c.speaker_party,
                    c.debate_title,
                    c.debate_date,
                    c.chunk_text,
                    uac.user_count,
                    uac.total_annotations,
                    STRING_AGG(DISTINCT ah.user_name, ', ') as annotators,
                    STRING_AGG(DISTINCT ah.frame_label, ', ') as frame_labels,
                    STRING_AGG(DISTINCT ah.brexit_position, ', ') as brexit_positions
                FROM user_annotation_counts uac
                JOIN chunks c ON uac.chunk_id = c.chunk_id
                JOIN annotation_history ah ON c.chunk_id = ah.chunk_id
                WHERE ah.frame_label IS NOT NULL AND ah.frame_label != ''
                GROUP BY c.chunk_id, c.speech_id, c.speaker_name, c.speaker_party, 
                         c.debate_title, c.debate_date, c.chunk_text, uac.user_count, uac.total_annotations
            )
            SELECT 
                COUNT(*) as total_double_classified,
                AVG(user_count) as avg_users_per_chunk,
                MAX(user_count) as max_users_per_chunk,
                COUNT(CASE WHEN user_count = 2 THEN 1 END) as double_classified,
                COUNT(CASE WHEN user_count = 3 THEN 1 END) as triple_classified,
                COUNT(CASE WHEN user_count > 3 THEN 1 END) as more_than_triple
            FROM double_classified_details
        """)
        
        stats = cursor.fetchone()
        
        # Zusätzliche Details für die Anzeige
        cursor.execute("""
            WITH user_annotation_counts AS (
                SELECT 
                    chunk_id,
                    COUNT(DISTINCT user_name) as user_count
                FROM annotation_history
                WHERE frame_label IS NOT NULL AND frame_label != ''
                GROUP BY chunk_id
                HAVING COUNT(DISTINCT user_name) > 1
            )
            SELECT 
                ah.user_name,
                COUNT(DISTINCT uac.chunk_id) as chunks_double_classified
            FROM user_annotation_counts uac
            JOIN annotation_history ah ON uac.chunk_id = ah.chunk_id
            WHERE ah.frame_label IS NOT NULL AND ah.frame_label != ''
            GROUP BY ah.user_name
            ORDER BY chunks_double_classified DESC
        """)
        
        user_stats = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return {
            'total_double_classified': stats[0] or 0,
            'avg_users_per_chunk': float(stats[1]) if stats[1] else 0,
            'max_users_per_chunk': stats[2] or 0,
            'double_classified': stats[3] or 0,
            'triple_classified': stats[4] or 0,
            'more_than_triple': stats[5] or 0,
            'by_user': {user: count for user, count in user_stats}
        }
        
    except Exception as e:
        st.error(f"Fehler beim Laden der Doppelklassifizierungs-Statistiken: {e}")
        if conn:
            conn.close()
        return {}

def show_conflict_resolution():
    """Zeigt Interface zur Konfliktlösung"""
    st.subheader("⚔️ Annotation-Konflikte")
    st.markdown("Übersicht über echte Konflikte zwischen verschiedenen Usern und deren Lösung")
    
    # Lade echte Konflikte
    conflicts = cached_get_annotation_conflicts()
    
    # Lade Doppelklassifizierungs-Statistiken
    double_stats = get_double_classification_stats()
    
    # Zeige Statistiken über doppelt klassifizierte Einträge
    if double_stats and double_stats.get('total_double_classified', 0) > 0:
        st.subheader("📊 Doppelklassifizierungs-Statistiken")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Doppelt klassifiziert", 
                f"{double_stats['total_double_classified']:,}",
                help="Chunks, die von mehreren Usern klassifiziert wurden"
            )
        
        with col2:
            st.metric(
                "Durchschnitt User/Chunk", 
                f"{double_stats['avg_users_per_chunk']:.1f}",
                help="Durchschnittliche Anzahl User pro doppelt klassifiziertem Chunk"
            )
        
        with col3:
            st.metric(
                "Max User/Chunk", 
                f"{double_stats['max_users_per_chunk']}",
                help="Maximale Anzahl User für einen einzelnen Chunk"
            )
        
        with col4:
            st.metric(
                "Nur 2 User", 
                f"{double_stats['double_classified']:,}",
                help="Chunks mit genau 2 klassifizierenden Usern"
            )
        
        # Detaillierte Aufschlüsselung
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Aufschlüsselung nach User-Anzahl:**")
            st.write(f"- 2 User: {double_stats['double_classified']:,} Chunks")
            st.write(f"- 3 User: {double_stats['triple_classified']:,} Chunks")
            st.write(f"- 4+ User: {double_stats['more_than_triple']:,} Chunks")
        
        with col2:
            if double_stats.get('by_user'):
                st.write("**Top User bei Doppelklassifizierungen:**")
                for user, count in list(double_stats['by_user'].items())[:5]:
                    st.write(f"- **{user}**: {count:,} Chunks")
        
        st.divider()
    
    if not conflicts:
        st.info("🎉 Keine Konflikte gefunden! Alle Annotationen sind konsistent.")
        
        # Zeige zusätzliche Informationen wenn keine Konflikte vorhanden
        st.subheader("📊 Annotation-Übersicht")
        
        # Lade Statistiken über alle Annotationen
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                
                # User-Statistiken
                cursor.execute("""
                    SELECT user_name, COUNT(DISTINCT chunk_id) as chunks
                    FROM (
                        SELECT assigned_user as user_name, chunk_id FROM chunks WHERE frame_label IS NOT NULL AND frame_label != ''
                        UNION ALL
                        SELECT user_name, chunk_id FROM annotation_history WHERE frame_label IS NOT NULL AND frame_label != ''
                    ) all_annotations
                    GROUP BY user_name
                    ORDER BY chunks DESC
                """)
                
                user_stats = cursor.fetchall()
                if user_stats:
                    st.write("**User-Statistiken:**")
                    for user, count in user_stats:
                        st.write(f"- **{user}**: {count:,} Chunks")
                
                # Frame-Verteilung
                cursor.execute("""
                    SELECT frame_label, COUNT(*) as count
                    FROM (
                        SELECT frame_label FROM chunks WHERE frame_label IS NOT NULL AND frame_label != ''
                        UNION ALL
                        SELECT frame_label FROM annotation_history WHERE frame_label IS NOT NULL AND frame_label != ''
                    ) all_frames
                    GROUP BY frame_label
                    ORDER BY count DESC
                """)
                
                frame_stats = cursor.fetchall()
                if frame_stats:
                    st.write("**Frame-Verteilung:**")
                    for frame, count in frame_stats:
                        st.write(f"- **{frame}**: {count:,} Annotationen")
                
                cursor.close()
                conn.close()
                
            except Exception as e:
                st.error(f"Fehler beim Laden der Statistiken: {e}")
                if conn:
                    conn.close()
        
        st.info("💡 **Hinweis:** Um Konflikte zu testen, könntest du einen bereits annotierten Chunk von einem anderen User nochmal annotieren.")
        return
    
    st.write(f"**Gefundene Konflikte:** {len(conflicts)}")
    
    # Filter-Optionen
    col1, col2, col3 = st.columns(3)
    
    with col1:
        frame_filter = st.selectbox(
            "Frame-Filter:",
            options=["Alle"] + FRAME_CATEGORIES,
            key="conflict_frame_filter"
        )
    
    with col2:
        user_filter = st.selectbox(
            "User-Filter:",
            options=["Alle"] + list(set([c['user1'] for c in conflicts] + [c['user2'] for c in conflicts])),
            key="conflict_user_filter"
        )
    
    with col3:
        brexit_filter = st.selectbox(
            "Brexit-Position Filter:",
            options=["Alle"] + BREXIT_POSITION_CATEGORIES,
            key="conflict_brexit_filter"
        )
    
    # Filtere Konflikte
    filtered_conflicts = conflicts
    
    if frame_filter != "Alle":
        filtered_conflicts = [c for c in filtered_conflicts if c['frame1'] == frame_filter or c['frame2'] == frame_filter]
    
    if user_filter != "Alle":
        filtered_conflicts = [c for c in filtered_conflicts if c['user1'] == user_filter or c['user2'] == user_filter]
    
    if brexit_filter != "Alle":
        filtered_conflicts = [c for c in filtered_conflicts if c['brexit1'] == brexit_filter or c['brexit2'] == brexit_filter]
    
    st.write(f"**Gefilterte Konflikte:** {len(filtered_conflicts)}")
    
    if not filtered_conflicts:
        st.info("Keine Konflikte entsprechen den Filterkriterien.")
        return
    
    # Zeige Konflikte
    for i, conflict in enumerate(filtered_conflicts):
        # Bestimme Konflikttyp (nur Frame-Konflikte)
        conflict_type = "Frame-Konflikt"
        
        with st.expander(f"Konflikt {i+1}: {conflict['original_chunk'][:20]}... - {conflict['frame1']} vs {conflict['frame2']} ({conflict_type})"):
            
            # Chunk-Informationen
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Speaker:** {conflict['speaker_name']} ({conflict['speaker_party']})")
            with col2:
                st.write(f"**Datum:** {conflict['debate_date']}")
            with col3:
                st.write(f"**Original:** {conflict['original_chunk']}<br/>**Duplikat:** {conflict['dup_chunk']}", unsafe_allow_html=True)
            
            # Chunk-Text
            st.subheader("📄 Chunk-Text")
            st.text_area("Chunk-Text", conflict['chunk_text'], height=150, disabled=True, key=f"text_{conflict['original_chunk']}", label_visibility="collapsed")
            
            # Beide Annotationen anzeigen
            st.subheader("⚔️ Konfliktierende Frame-Annotationen")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 👤 **User 1: " + conflict['user1'] + "**")
                st.write(f"**Chunk:** {conflict['original_chunk']}")
                st.write(f"**Frame:** {conflict['frame1']}")
                st.write(f"**Notizen:** {conflict['notes1'] or 'Keine Notizen'}")
            
            with col2:
                st.markdown("### 👤 **User 2: " + conflict['user2'] + "**")
                st.write(f"**Chunk:** {conflict['dup_chunk']}")
                st.write(f"**Frame:** {conflict['frame2']}")
                st.write(f"**Notizen:** {conflict['notes2'] or 'Keine Notizen'}")
            
            # Konfliktlösung
            st.subheader("🔧 Konfliktlösung")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Finale Frame-Annotation:**")
                new_frame = st.selectbox(
                    "Frame-Kategorie:",
                    options=FRAME_CATEGORIES,
                    index=FRAME_CATEGORIES.index(conflict['frame1']) if conflict['frame1'] in FRAME_CATEGORIES else 0,
                    key=f"new_frame_{conflict['original_chunk']}"
                )
            
            with col2:
                new_notes = st.text_area(
                    "Notizen:",
                    value=f"Konflikt gelöst zwischen {conflict['user1']} und {conflict['user2']}",
                    key=f"new_notes_{conflict['original_chunk']}"
                )
                
                resolved_by = st.text_input(
                    "Gelöst von:",
                    value=st.session_state.user_name or "",
                    key=f"resolved_by_{conflict['original_chunk']}"
                )
            
            # Schnellauswahl-Buttons für häufige Entscheidungen
            st.subheader("⚡ Schnellauswahl")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button(f"✅ {conflict['frame1']}", key=f"quick1_{conflict['original_chunk']}"):
                    st.session_state[f"new_frame_{conflict['original_chunk']}"] = conflict['frame1']
                    safe_rerun()
            
            with col2:
                if st.button(f"✅ {conflict['frame2']}", key=f"quick2_{conflict['original_chunk']}"):
                    st.session_state[f"new_frame_{conflict['original_chunk']}"] = conflict['frame2']
                    safe_rerun()
            
            with col3:
                if st.button("🔄 Aktualisieren", key=f"refresh_{conflict['original_chunk']}"):
                    safe_rerun()
            
            with col4:
                if st.button("🗑️ Beide entfernen", key=f"delete_{conflict['original_chunk']}"):
                    if st.session_state.user_name:
                        success = resolve_conflict(
                            conflict['original_chunk'], 
                            None, 
                            None, 
                            f"Beide Annotationen entfernt von {st.session_state.user_name}", 
                            st.session_state.user_name
                        )
                        if success:
                            st.success("✅ Beide Annotationen entfernt!")
                            safe_rerun()
                        else:
                            st.error("❌ Fehler beim Entfernen der Annotationen!")
                    else:
                        st.error("Bitte gib deinen Namen ein!")
            
            # Finale Lösung
            st.subheader("🎯 Finale Lösung")
            if st.button("✅ Konflikt endgültig lösen", key=f"resolve_{conflict['original_chunk']}", type="primary"):
                if new_frame and resolved_by:
                    success = resolve_conflict(
                        conflict['original_chunk'], 
                        new_frame, 
                        None,  # Keine Brexit-Position für Frame-Konflikte
                        new_notes, 
                        resolved_by
                    )
                    if success:
                        st.success("✅ Konflikt erfolgreich gelöst!")
                        safe_rerun()
                    else:
                        st.error("❌ Fehler beim Lösen des Konflikts!")
                else:
                    st.error("Bitte fülle alle erforderlichen Felder aus!")

def show_classified_chunks():
    """Zeigt alle klassifizierten Chunks in einer filterbaren Tabelle"""
    st.subheader("📋 Klassifizierte Chunks")
    st.markdown("Übersicht über alle bereits klassifizierten Chunks mit Filter- und Suchoptionen")
    
    # Lade klassifizierte Chunks
    chunks = cached_get_classified_chunks()
    
    if not chunks:
        st.info("Keine klassifizierten Chunks gefunden.")
        return
    
    st.write(f"**Gesamt klassifizierte Chunks:** {len(chunks)}")
    
    # Erstelle DataFrame für bessere Darstellung (alle Chunks)
    display_data = []
    for chunk in chunks:
        display_data.append({
            'Chunk-ID': chunk['chunk_id'][:20] + '...' if len(chunk['chunk_id']) > 20 else chunk['chunk_id'],
            'Speaker': chunk['speaker_name'],
            'Party': chunk['speaker_party'],
            'Frame': chunk['frame_label'],
            'Brexit-Pos': chunk['brexit_position'] or 'N/A',
            'User': chunk['assigned_user'],
            'Wörter': chunk['word_count'],
            'Datum': chunk['debate_date'],
            'Pre-Brexit': 'Ja' if chunk['pre_brexit'] else 'Nein',
            'Notizen': chunk['annotation_notes'][:50] + '...' if chunk['annotation_notes'] and len(chunk['annotation_notes']) > 50 else chunk['annotation_notes'] or ''
        })
    
    df = pd.DataFrame(display_data)
    
    # Zeige Tabelle (alle Chunks)
    st.subheader("📊 Alle klassifizierten Chunks")
    
    # Sortierbare Tabelle
    st.dataframe(
        df,
        width='stretch',
        height=600
    )
    
    # Filter-Optionen (optional)
    with st.expander("🔍 Filter & Suche (optional)"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            frame_filter = st.selectbox(
                "Frame-Kategorie:",
                options=["Alle"] + FRAME_CATEGORIES,
                key="classified_frame_filter"
            )
        
        with col2:
            user_filter = st.selectbox(
                "User:",
                options=["Alle"] + list(set([c['assigned_user'] for c in chunks if c['assigned_user']])),
                key="classified_user_filter"
            )
        
        with col3:
            brexit_filter = st.selectbox(
                "Brexit-Position:",
                options=["Alle"] + BREXIT_POSITION_CATEGORIES,
                key="classified_brexit_filter"
            )
        
        with col4:
            speaker_filter = st.selectbox(
                "Speaker:",
                options=["Alle"] + list(set([c['speaker_name'] for c in chunks if c['speaker_name']])),
                key="classified_speaker_filter"
            )
        
        # Zusätzliche Filter
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pre_brexit_filter = st.selectbox(
                "Brexit-Timing:",
                options=["Alle", "Pre-Brexit", "Post-Brexit"],
                key="classified_timing_filter"
            )
        
        with col2:
            search_text = st.text_input(
                "Text-Suche:",
                placeholder="Suche in Chunk-Text...",
                key="classified_text_search"
            )
        
        with col3:
            date_from = st.date_input(
                "Von Datum:",
                key="classified_date_from"
            )
        
        # Filtere Chunks
        filtered_chunks = chunks
        
        if frame_filter != "Alle":
            filtered_chunks = [c for c in filtered_chunks if c['frame_label'] == frame_filter]
        
        if user_filter != "Alle":
            filtered_chunks = [c for c in filtered_chunks if c['assigned_user'] == user_filter]
        
        if brexit_filter != "Alle":
            filtered_chunks = [c for c in filtered_chunks if c['brexit_position'] == brexit_filter]
        
        if speaker_filter != "Alle":
            filtered_chunks = [c for c in filtered_chunks if c['speaker_name'] == speaker_filter]
        
        if pre_brexit_filter == "Pre-Brexit":
            filtered_chunks = [c for c in filtered_chunks if c['pre_brexit'] == True]
        elif pre_brexit_filter == "Post-Brexit":
            filtered_chunks = [c for c in filtered_chunks if c['pre_brexit'] == False]
        
        if search_text:
            filtered_chunks = [c for c in filtered_chunks if search_text.lower() in c['chunk_text'].lower()]
        
        if date_from:
            filtered_chunks = [c for c in filtered_chunks if c['debate_date'] >= date_from]
        
        st.write(f"**Gefilterte Chunks:** {len(filtered_chunks)}")
        
        if len(filtered_chunks) != len(chunks):
            # Zeige gefilterte Tabelle
            filtered_display_data = []
            for chunk in filtered_chunks:
                filtered_display_data.append({
                    'Chunk-ID': chunk['chunk_id'][:20] + '...' if len(chunk['chunk_id']) > 20 else chunk['chunk_id'],
                    'Speaker': chunk['speaker_name'],
                    'Party': chunk['speaker_party'],
                    'Frame': chunk['frame_label'],
                    'Brexit-Pos': chunk['brexit_position'] or 'N/A',
                    'User': chunk['assigned_user'],
                    'Wörter': chunk['word_count'],
                    'Datum': chunk['debate_date'],
                    'Pre-Brexit': 'Ja' if chunk['pre_brexit'] else 'Nein',
                    'Notizen': chunk['annotation_notes'][:50] + '...' if chunk['annotation_notes'] and len(chunk['annotation_notes']) > 50 else chunk['annotation_notes'] or ''
                })
            
            filtered_df = pd.DataFrame(filtered_display_data)
            st.dataframe(filtered_df, width='stretch', height=400)
    
    # Aktualisieren-Button
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Aktualisieren"):
            safe_rerun()
    
    # Detaillierte Ansicht für ausgewählte Chunks
    st.subheader("🔍 Detaillierte Ansicht")
    
    selected_chunk_id = st.selectbox(
        "Wähle einen Chunk für Details:",
        options=[c['chunk_id'] for c in chunks],
        format_func=lambda x: f"{x[:20]}... - {next(c['frame_label'] for c in chunks if c['chunk_id'] == x)}",
        key="detailed_chunk_selector"
    )
    
    if selected_chunk_id:
        selected_chunk = next(c for c in chunks if c['chunk_id'] == selected_chunk_id)
        
        st.write("**Chunk-Details:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Chunk-ID:** {selected_chunk['chunk_id']}")
            st.write(f"**Speaker:** {selected_chunk['speaker_name']} ({selected_chunk['speaker_party']})")
            st.write(f"**Debatte:** {selected_chunk['debate_title']}")
            st.write(f"**Datum:** {selected_chunk['debate_date']}")
            st.write(f"**Pre-Brexit:** {'Ja' if selected_chunk['pre_brexit'] else 'Nein'}")
        
        with col2:
            st.write(f"**Frame:** {selected_chunk['frame_label']}")
            st.write(f"**Brexit-Position:** {selected_chunk['brexit_position'] or 'Nicht gesetzt'}")
            st.write(f"**User:** {selected_chunk['assigned_user']}")
            st.write(f"**Wörter:** {selected_chunk['word_count']}")
            st.write(f"**Zeichen:** {selected_chunk['char_count']}")
        
        st.write("**Chunk-Text:**")
        st.text_area("Chunk-Text", selected_chunk['chunk_text'], height=200, disabled=True, label_visibility="collapsed")
        
        if selected_chunk['annotation_notes']:
            st.write("**Notizen:**")
            st.write(selected_chunk['annotation_notes'])

def show_admin_view():
    """Zeigt Admin-Ansicht"""
    st.subheader("👥 Admin-Ansicht")
    
    conn = get_db_connection()
    if not conn:
        st.error("Keine Datenbankverbindung!")
        return
    
    try:
        cursor = conn.cursor()
        
        # Alle Zuweisungen
        cursor.execute("""
            SELECT assigned_user, COUNT(*) as total_chunks,
                   COUNT(CASE WHEN frame_label IS NOT NULL THEN 1 END) as annotated_chunks
            FROM chunks 
            WHERE assigned_user IS NOT NULL AND assigned_user != ''
            GROUP BY assigned_user
            ORDER BY assigned_user
        """)
        assignments = cursor.fetchall()
        
        if assignments:
            st.write("**Chunk-Zuweisungen:**")
            
            # Erstelle DataFrame für bessere Darstellung
            admin_data = []
            for user, total, annotated in assignments:
                admin_data.append({
                    'User': user,
                    'Zugewiesene Chunks': total,
                    'Annotierte Chunks': annotated,
                    'Fortschritt': f"{annotated}/{total}",
                    'Prozent': f"{(annotated/total*100):.1f}%" if total > 0 else "0%"
                })
            
            df = pd.DataFrame(admin_data)
            st.dataframe(df, width='stretch')
        
        # Unzugewiesene Chunks
        cursor.execute("""
            SELECT COUNT(*) FROM chunks 
            WHERE assigned_user IS NULL OR assigned_user = ''
        """)
        unassigned = cursor.fetchone()[0]
        
        st.metric("Unzugewiesene Chunks", f"{unassigned:,}")
        
        # Gesamt-Statistiken
        cursor.execute("""
            SELECT 
                COUNT(*) as total_chunks,
                COUNT(CASE WHEN frame_label IS NOT NULL THEN 1 END) as annotated_chunks
            FROM chunks
        """)
        total_stats = cursor.fetchone()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Gesamt Chunks", f"{total_stats[0]:,}")
        with col2:
            st.metric("Annotierte Chunks", f"{total_stats[1]:,}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        st.error(f"Fehler beim Laden der Admin-Daten: {e}")
        if conn:
            conn.close()

def main():
    st.set_page_config(
        page_title="Frame Classification - Railway",
        page_icon="🏷️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # WebSocket-Fehler reduzieren durch optimierte Reruns
    if 'last_rerun' not in st.session_state:
        st.session_state.last_rerun = 0
    
    st.title("🏷️ Frame Classification - Railway")
    st.markdown("**Multi-User Annotation von Brexit-Debatten Chunks auf Railway**")
    
    # Debug-Informationen (nur in Development und reduziert)
    if os.getenv('RAILWAY_ENVIRONMENT') != 'production' and os.getenv('DEBUG', 'false').lower() == 'true':
        with st.expander("🔧 Debug-Informationen"):
            st.write(f"DATABASE_URL gesetzt: {'Ja' if os.getenv('DATABASE_URL') else 'Nein'}")
            if os.getenv('DATABASE_URL'):
                # Zeige nur den Anfang der URL für Sicherheit
                db_url = os.getenv('DATABASE_URL')
                st.write(f"DATABASE_URL: {db_url[:20]}...")
            else:
                st.write("Verwende lokale Konfiguration")
    
    # Initialisiere Session State
    init_session_state()
    
    # Erstelle Tabellen falls nötig
    create_tables_if_not_exist()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Konfiguration")
        
        # User-Name
        user_name = st.text_input(
            "👤 Dein Name:",
            value=st.session_state.user_name,
            help="Wird für die Zuweisung von Chunks verwendet"
        )
        if user_name != st.session_state.user_name:
            st.session_state.user_name = user_name
            safe_rerun()
        
        # Chunk-Limit
        chunk_limit = st.number_input(
            "📊 Chunk-Limit:",
            min_value=10,
            max_value=1000,
            value=100,
            help="Anzahl der Chunks zum Laden"
        )
        
        # Chunk-Typ Auswahl
        chunk_type = st.radio(
            "📝 Chunk-Typ:",
            options=["Nur unklassifizierte", "Alle Chunks"],
            help="Wähle ob nur unklassifizierte Chunks oder alle Chunks geladen werden sollen"
        )
        only_unannotated = chunk_type == "Nur unklassifizierte"
        
        # Lade Chunks
        if st.button("🔄 Chunks laden"):
            if not st.session_state.user_name:
                st.error("❌ Bitte gib zuerst deinen Namen ein!")
            else:
                with st.spinner("🔄 Lade Chunks aus PostgreSQL..."):
                    try:
                        st.session_state.chunks = load_database_chunks(st.session_state.user_name, chunk_limit, only_unannotated)
                        st.session_state.current_chunk_index = 0
                        
                        chunk_type_text = "unklassifizierte" if only_unannotated else "alle"
                        if st.session_state.chunks:
                            st.success(f"✅ {len(st.session_state.chunks)} {chunk_type_text} Chunks für {st.session_state.user_name} geladen!")
                        else:
                            st.warning(f"⚠️ Keine {chunk_type_text} Chunks für {st.session_state.user_name} gefunden!")
                    except Exception as e:
                        st.error(f"❌ Fehler beim Laden der Chunks: {e}")
                        st.error("🔧 Bitte versuche es erneut oder kontaktiere den Administrator")
        
        st.divider()
        
        # Statistiken
        if st.button("📊 Statistiken aktualisieren"):
            safe_rerun()
    
    # Hauptbereich
    if not st.session_state.chunks:
        st.info("👆 Lade zuerst Chunks aus der Datenbank!")
        st.info("💡 **Tipp:** Wähle 'Nur unklassifizierte' um nur noch zu klassifizierende Chunks zu laden.")
        return
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Annotation", "📊 Statistiken", "👥 Admin", "⚔️ Konflikte", "📋 Klassifiziert"])
    
    with tab1:
        show_chunk_annotation()
    
    with tab2:
        show_statistics()
    
    with tab3:
        show_admin_view()
    
    with tab4:
        show_conflict_resolution()
    
    with tab5:
        show_classified_chunks()

if __name__ == "__main__":
    main()
