#!/bin/bash
# Quick start script for Multi-Team Response Comparison Dashboard

echo "🚀 Starting Multi-Team Response Comparison Dashboard..."
echo ""
echo "The dashboard will open in your browser at http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

cd "$(dirname "$0")"
streamlit run app.py --server.headless=false

