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

# Railway PostgreSQL Konfiguration (Fallback für lokale Entwicklung)
DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'frame_classification'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '')
}

# Frame-Kategorien
FRAME_CATEGORIES = [
    "Human Impact",
    "Powerlessness", 
    "Economic",
    "Moral Value",
    "Conflict",
    "Other"
]

def get_db_connection():
    """Erstellt PostgreSQL Verbindung für Railway"""
    try:
        # Railway DATABASE_URL Format: postgresql://user:password@host:port/database
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            # Verwende die DATABASE_URL direkt
            conn = psycopg2.connect(database_url)
            return conn
        else:
            # Fallback für lokale Entwicklung
            conn = psycopg2.connect(**DATABASE_CONFIG)
            return conn
    except Exception as e:
        st.error(f"Fehler bei Datenbankverbindung: {e}")
        st.error(f"DATABASE_URL: {os.getenv('DATABASE_URL', 'Nicht gesetzt')}")
        return None

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
        
        # Indizes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_assigned_user ON chunks(assigned_user);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_frame_label ON chunks(frame_label);")
        
        conn.commit()
        cursor.close()
        conn.close()
        
    except Exception as e:
        st.error(f"Fehler beim Erstellen der Tabellen: {e}")
        if conn:
            conn.close()

def load_database_chunks(user_name: str = None, limit: int = None) -> List[Dict[str, Any]]:
    """Lädt Chunks aus PostgreSQL für einen bestimmten User"""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        if user_name:
            # Lade nur Chunks für den spezifischen User
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
        conn.close()
        
        # Konvertiere zu Dictionary-Liste
        chunk_list = []
        for chunk in chunks:
            chunk_dict = dict(chunk)
            chunk_list.append(chunk_dict)
        
        return chunk_list
        
    except Exception as e:
        st.error(f"Fehler beim Laden der Chunks: {e}")
        if conn:
            conn.close()
        return []

def update_database_annotation(chunk_id: str, frame_label: str, confidence: int, notes: str, user_name: str):
    """Aktualisiert Annotation in PostgreSQL"""
    conn = get_db_connection()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        
        update_sql = """
        UPDATE chunks 
        SET frame_label = %s, annotation_confidence = %s, annotation_notes = %s, 
            assigned_user = %s, updated_at = CURRENT_TIMESTAMP
        WHERE chunk_id = %s
        """
        
        cursor.execute(update_sql, (frame_label, confidence, notes, user_name, chunk_id))
        conn.commit()
        cursor.close()
        conn.close()
        
    except Exception as e:
        st.error(f"Fehler beim Aktualisieren der Datenbank: {e}")
        if conn:
            conn.rollback()
            conn.close()

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
        
        cursor.close()
        conn.close()
        
        return {
            'total_chunks': total_stats[0],
            'annotated_chunks': total_stats[1],
            'assigned_chunks': total_stats[2],
            'by_frame': {frame: count for frame, count in frame_stats},
            'by_user': {user: count for user, count in user_stats}
        }
        
    except Exception as e:
        st.error(f"Fehler beim Laden der Statistiken: {e}")
        if conn:
            conn.close()
        return {}

def show_statistics():
    """Zeigt Statistiken"""
    stats = get_statistics()
    
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
    
    # Chunk-Text
    st.subheader("📄 Chunk-Text")
    st.text_area("", current_chunk['chunk_text'], height=200, disabled=True)
    
    # Annotation-Formular
    st.subheader("🏷️ Frame-Annotation")
    
    # Frame-Auswahl
    frame_label = st.selectbox(
        "Frame-Kategorie:",
        options=[""] + FRAME_CATEGORIES,
        key=f"frame_{chunk_id}"
    )
    
    # Notes
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
                # Aktualisiere Datenbank
                update_database_annotation(
                    chunk_id, frame_label, 3, notes, st.session_state.user_name
                )
                
                st.success("✅ Annotation gespeichert!")
                
                # Automatisch zum nächsten Chunk
                if st.session_state.current_chunk_index < len(st.session_state.chunks) - 1:
                    st.session_state.current_chunk_index += 1
                    # Formularfelder zurücksetzen
                    if f"frame_{chunk_id}" in st.session_state:
                        del st.session_state[f"frame_{chunk_id}"]
                    if f"notes_{chunk_id}" in st.session_state:
                        del st.session_state[f"notes_{chunk_id}"]
                    st.info("🔄 Lade nächsten Chunk...")
                else:
                    st.info("🎉 Alle Chunks annotiert!")
                
                st.rerun()
            else:
                st.error("Bitte wähle eine Frame-Kategorie!")
    
    with col2:
        if st.button("⏭️ Nächster"):
            if st.session_state.current_chunk_index < len(st.session_state.chunks) - 1:
                st.session_state.current_chunk_index += 1
                st.rerun()
            else:
                st.info("Letzter Chunk erreicht!")
    
    with col3:
        if st.button("⏮️ Vorheriger"):
            if st.session_state.current_chunk_index > 0:
                st.session_state.current_chunk_index -= 1
                st.rerun()
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
        st.rerun()

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
            st.dataframe(df, use_container_width=True)
        
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
        layout="wide"
    )
    
    st.title("🏷️ Frame Classification - Railway")
    st.markdown("**Multi-User Annotation von Brexit-Debatten Chunks auf Railway**")
    
    # Debug-Informationen (nur in Development)
    if os.getenv('RAILWAY_ENVIRONMENT') != 'production':
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
            st.rerun()
        
        # Chunk-Limit
        chunk_limit = st.number_input(
            "📊 Chunk-Limit:",
            min_value=10,
            max_value=1000,
            value=100,
            help="Anzahl der Chunks zum Laden"
        )
        
        # Lade Chunks
        if st.button("🔄 Chunks laden"):
            if not st.session_state.user_name:
                st.error("Bitte gib zuerst deinen Namen ein!")
            else:
                with st.spinner("Lade Chunks aus PostgreSQL..."):
                    st.session_state.chunks = load_database_chunks(st.session_state.user_name, chunk_limit)
                    st.session_state.current_chunk_index = 0
                st.success(f"✓ {len(st.session_state.chunks)} Chunks für {st.session_state.user_name} geladen!")
        
        st.divider()
        
        # Statistiken
        if st.button("📊 Statistiken aktualisieren"):
            st.rerun()
    
    # Hauptbereich
    if not st.session_state.chunks:
        st.info("👆 Lade zuerst Chunks aus der Datenbank!")
        return
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Annotation", "📊 Statistiken", "👥 Admin"])
    
    with tab1:
        show_chunk_annotation()
    
    with tab2:
        show_statistics()
    
    with tab3:
        show_admin_view()

if __name__ == "__main__":
    main()
