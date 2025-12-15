#!/bin/bash
# Railway Start Script - Python Version
echo "Starting Railway deployment with Python..."

# Debug information
echo "Python version:"
/opt/venv/bin/python --version

echo "Installed packages:"
/opt/venv/bin/pip list | grep streamlit

# Run streamlit as Python module
echo "Running streamlit as Python module..."
/opt/venv/bin/python -m streamlit run streamlit_annotation_railway.py --server.port=$PORT --server.address=0.0.0.0
