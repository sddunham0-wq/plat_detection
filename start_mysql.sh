#!/bin/bash

echo "=================================="
echo "🔧 MySQL Quick Start Script"
echo "=================================="
echo ""

# Check if MySQL is already running
if lsof -i :3306 > /dev/null 2>&1; then
    echo "✅ MySQL is already running on port 3306"
    echo ""
    mysql -V
    exit 0
fi

echo "🔍 MySQL not running. Attempting to start..."
echo ""

# Try Homebrew
if command -v brew &> /dev/null; then
    echo "📦 Trying Homebrew..."
    brew services start mysql
    sleep 3

    if lsof -i :3306 > /dev/null 2>&1; then
        echo "✅ MySQL started successfully via Homebrew!"
        exit 0
    fi
fi

# Try system service
if command -v systemctl &> /dev/null; then
    echo "🔧 Trying systemctl..."
    sudo systemctl start mysql
    sleep 3

    if lsof -i :3306 > /dev/null 2>&1; then
        echo "✅ MySQL started successfully via systemctl!"
        exit 0
    fi
fi

# Try manual start
if [ -f /usr/local/mysql/support-files/mysql.server ]; then
    echo "🔧 Trying manual start..."
    sudo /usr/local/mysql/support-files/mysql.server start
    sleep 3

    if lsof -i :3306 > /dev/null 2>&1; then
        echo "✅ MySQL started successfully!"
        exit 0
    fi
fi

echo ""
echo "❌ Failed to start MySQL automatically"
echo ""
echo "Please start MySQL manually:"
echo "  - XAMPP/MAMP: Open control panel and click Start"
echo "  - Command: brew services start mysql"
echo "  - Or: sudo systemctl start mysql"
echo ""
