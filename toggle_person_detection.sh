#!/bin/bash
# Quick toggle person detection via API

if [ "$1" == "on" ] || [ "$1" == "enable" ] || [ "$1" == "1" ]; then
    echo "👤 Enabling person detection..."
    curl -X POST http://localhost:5000/api/toggle_person_detection \
      -H "Content-Type: application/json" \
      -d '{"enable": true}' \
      -s | python3 -m json.tool

elif [ "$1" == "off" ] || [ "$1" == "disable" ] || [ "$1" == "0" ]; then
    echo "⏸️  Disabling person detection..."
    curl -X POST http://localhost:5000/api/toggle_person_detection \
      -H "Content-Type: application/json" \
      -d '{"enable": false}' \
      -s | python3 -m json.tool

elif [ "$1" == "status" ] || [ "$1" == "check" ]; then
    echo "📊 Checking person detection status..."
    curl -s http://localhost:5000/api/stats | python3 -c "
import sys, json
data = json.load(sys.stdin)
enabled = data.get('person_detection_enabled', False)
total = data.get('total_persons_detected', 0)
print(f'Person Detection: {\"✅ ENABLED\" if enabled else \"⏸️ DISABLED\"}')
print(f'Total Persons Detected: {total}')
"

else
    echo "Usage: $0 [on|off|status]"
    echo ""
    echo "Examples:"
    echo "  $0 on       # Enable person detection"
    echo "  $0 off      # Disable person detection"
    echo "  $0 status   # Check current status"
    echo ""
fi
