#!/bin/bash
# Start headless stream dengan person detection ENABLED

echo "🔄 Starting headless stream with PERSON DETECTION enabled..."

# Kill existing process on port 5000
PID=$(lsof -ti:5000 2>/dev/null)
if [ ! -z "$PID" ]; then
    echo "🛑 Killing existing process: PID $PID"
    kill -9 $PID 2>/dev/null
    sleep 1
fi

# Check if person detection is enabled in config
if grep -q "ENABLE_PERSON_DETECTION = False" config.py; then
    echo "⚠️  WARNING: Person detection is DISABLED in config.py"
    echo "   To enable permanently, edit config.py:"
    echo "   ENABLE_PERSON_DETECTION = True"
    echo ""
    echo "   Or use API to enable after starting:"
    echo "   curl -X POST http://localhost:5000/api/toggle_person_detection -H 'Content-Type: application/json' -d '{\"enable\": true}'"
    echo ""
fi

echo "✅ Port 5000 is free"
echo ""
echo "🚀 Starting server..."
echo "   📍 URL: http://localhost:5000"
echo "   👤 Person Detection: Check API to toggle"
echo "   🚗 Plate Detection: Always ON"
echo ""
echo "📊 To check person detection status:"
echo "   curl http://localhost:5000/api/stats | grep person_detection_enabled"
echo ""
echo "Press CTRL+C to stop"
echo ""

python headless_stream.py
