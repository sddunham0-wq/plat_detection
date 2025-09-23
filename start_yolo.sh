#!/bin/bash

# Quick Start Script untuk CCTV Detection dengan YOLOv8
# Usage: ./start_yolo.sh

echo "🎥 CCTV Live Detection - Quick Start with YOLOv8"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "headless_stream.py" ]; then
    echo "❌ Please run this script from the project directory"
    echo "   cd /path/to/project-plat-detection-alfi"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Start server with optimized settings
echo "🚀 Starting server with background YOLOv8 loading..."
echo ""
echo "🎯 Server will be available at: http://localhost:5000"
echo "⚡ YOLOv8 will auto-enable when ready"
echo "📹 Start your stream and watch object detection activate!"
echo ""
echo "Press Ctrl+C to stop"
echo "=================================================="

# Run the optimized startup
python3 start_with_yolo.py