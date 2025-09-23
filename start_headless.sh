#!/bin/bash
# Quick start script untuk headless CCTV system

echo "🚀 Starting Headless CCTV Detection System..."
echo "🌐 Web interface will be available at: http://localhost:5000"
echo "⏹️  Press Ctrl+C to stop"
echo ""

cd "$(dirname "$0")"

# Auto-start dengan RTSP default
python3 headless_stream.py --source "rtsp://admin:H4nd4l9165!@168.1.195:554"