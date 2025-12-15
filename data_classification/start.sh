#!/bin/bash
# Railway Start Script - Optimized
echo "Starting Railway deployment..."

# Install streamlit if not present
echo "Checking for streamlit..."
/opt/venv/bin/python -c "import streamlit" 2>/dev/null || {
    echo "Streamlit not found, installing..."
    /opt/venv/bin/pip install streamlit==1.50.0 plotly==6.3.1 pandas==2.3.3
}

# Activate virtual environment
source /opt/venv/bin/activate
echo "Virtual environment activated"

# Run streamlit with optimized settings
echo "Starting streamlit with optimized settings..."
/opt/venv/bin/python -m streamlit run streamlit_annotation_railway.py \
  --server.port=$PORT \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.runOnSave=false \
  --server.maxUploadSize=200 \
  --server.maxMessageSize=200 \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false \
  --browser.gatherUsageStats=false