#!/usr/bin/env python3
"""
Railway Start Script - Python Version
Startet Streamlit für Railway Deployment
"""

import os
import sys
import subprocess

def install_streamlit():
    """Installiert streamlit falls es nicht vorhanden ist"""
    try:
        import streamlit
        print("Streamlit bereits installiert")
        return True
    except ImportError:
        print("Streamlit nicht gefunden, installiere...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'streamlit'], check=True)
            print("Streamlit erfolgreich installiert")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Fehler beim Installieren von Streamlit: {e}")
            return False

def main():
    print("Starting Streamlit for Railway...")
    
    # Installiere streamlit falls nötig
    if not install_streamlit():
        print("Konnte Streamlit nicht installieren")
        sys.exit(1)
    
    # Set environment variables
    port = os.getenv('PORT', '8000')
    print(f"Using port: {port}")
    
    # Run streamlit
    cmd = [
        sys.executable, '-m', 'streamlit', 'run',
        'streamlit_annotation_railway.py',
        '--server.port', port,
        '--server.address', '0.0.0.0'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
