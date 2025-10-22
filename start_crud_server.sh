#!/bin/bash

# CCTV Access Control CRUD Server Starter
# Quick start script untuk Flask server dengan CRUD interface

echo "=================================="
echo "🚗 CCTV ACCESS CONTROL CRUD SERVER"
echo "=================================="
echo ""

# Check if MySQL is running
echo "🔍 Checking MySQL connection..."
python3 -c "from mysql_database import MySQLPlateDatabase; db = MySQLPlateDatabase(); db.test_connection()" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ MySQL connection OK"
else
    echo "❌ MySQL connection failed!"
    echo "   Please check your .env configuration"
    echo "   and ensure MySQL server is running."
    exit 1
fi

echo ""
echo "🌐 Starting Flask server..."
echo "=================================="
echo ""
echo "📱 Access web interface:"
echo "   Live Stream:  http://localhost:5010/"
echo "   Vehicles:     http://localhost:5010/vehicles"
echo "   Access Log:   http://localhost:5010/access-log"
echo ""
echo "🔑 DELETE PIN: 1234"
echo ""
echo "⏹️  Press Ctrl+C to stop"
echo "=================================="
echo ""

# Start Flask server
python3 headless_stream.py --port 5010
