#!/bin/bash
# Auto-restart script for headless_stream.py
# Akan auto-kill process yang sedang berjalan sebelum start

echo "🔄 Restarting headless stream server..."

# Kill existing process on port 5000
echo "🛑 Checking for existing process on port 5000..."
PID=$(lsof -ti:5000 2>/dev/null)

if [ ! -z "$PID" ]; then
    echo "   Found process: PID $PID"
    kill -9 $PID 2>/dev/null
    sleep 1
    echo "   ✅ Process killed"
else
    echo "   ℹ️  No existing process found"
fi

# Verify port is free
if lsof -ti:5000 >/dev/null 2>&1; then
    echo "❌ Error: Port 5000 still in use!"
    exit 1
else
    echo "✅ Port 5000 is free"
fi

# Start headless stream
echo ""
echo "🚀 Starting headless stream server..."
echo "   URL: http://localhost:5000"
echo "   Press CTRL+C to stop"
echo ""

python headless_stream.py

# If script exits, show message
echo ""
echo "⏹️  Server stopped"
