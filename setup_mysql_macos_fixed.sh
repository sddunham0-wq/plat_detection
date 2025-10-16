#!/bin/bash
# ============================================================
# MYSQL SETUP SCRIPT FOR macOS
# Auto-install & configure MySQL untuk macOS server
# ============================================================

set -e  # Exit on error

echo "============================================================"
echo "  MySQL Setup for macOS - Sistem Parkir SMK"
echo "============================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check if Homebrew installed
echo "📋 Step 1: Checking Homebrew..."
if ! command -v brew &> /dev/null; then
    echo -e "${YELLOW}⚠️  Homebrew not found. Installing Homebrew...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Add Homebrew to PATH for Apple Silicon Macs
    if [[ $(uname -m) == 'arm64' ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi

    echo -e "${GREEN}✅ Homebrew installed successfully!${NC}"
else
    echo -e "${GREEN}✅ Homebrew already installed${NC}"
fi

echo ""

# Step 2: Install MySQL
echo "📋 Step 2: Installing MySQL..."
if brew list mysql &> /dev/null; then
    echo -e "${GREEN}✅ MySQL already installed${NC}"
else
    echo -e "${YELLOW}⚙️  Installing MySQL via Homebrew...${NC}"
    brew install mysql
    echo -e "${GREEN}✅ MySQL installed successfully!${NC}"
fi

echo ""

# Step 3: Start MySQL service
echo "📋 Step 3: Starting MySQL service..."
brew services start mysql

# Wait for MySQL to start
echo "⏳ Waiting for MySQL to start (5 seconds)..."
sleep 5

# Check if MySQL is running
if brew services list | grep -q "mysql.*started"; then
    echo -e "${GREEN}✅ MySQL service is running!${NC}"
else
    echo -e "${RED}❌ MySQL failed to start${NC}"
    echo "Try manually: brew services start mysql"
    exit 1
fi

echo ""

# Step 4: Check MySQL version
echo "📋 Step 4: Checking MySQL version..."
mysql --version

echo ""

# Step 5: Create database
echo "📋 Step 5: Creating database 'sistem_parkir_smk'..."

# Check if database exists
DB_EXISTS=$(mysql -u root -e "SHOW DATABASES LIKE 'sistem_parkir_smk';" 2>/dev/null | grep -c "sistem_parkir_smk" || true)

if [ "$DB_EXISTS" -eq 1 ]; then
    echo -e "${YELLOW}⚠️  Database 'sistem_parkir_smk' already exists${NC}"
    read -p "Drop and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        mysql -u root -e "DROP DATABASE sistem_parkir_smk;"
        echo -e "${GREEN}✅ Old database dropped${NC}"
    else
        echo "Skipping database creation..."
        echo ""
        echo "============================================================"
        echo -e "${GREEN}✅ MySQL Setup Complete!${NC}"
        echo "============================================================"
        exit 0
    fi
fi

# Create database
mysql -u root -e "CREATE DATABASE sistem_parkir_smk CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
echo -e "${GREEN}✅ Database 'sistem_parkir_smk' created!${NC}"

echo ""

# Step 6: Import schema
echo "📋 Step 6: Importing database schema..."
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SQL_FILE="${SCRIPT_DIR}/database_setup.sql"

if [ -f "$SQL_FILE" ]; then
    mysql -u root sistem_parkir_smk < "$SQL_FILE"
    echo -e "${GREEN}✅ Database schema imported successfully!${NC}"
else
    echo -e "${RED}❌ Error: database_setup.sql not found!${NC}"
    echo "Looking for: $SQL_FILE"
    exit 1
fi

echo ""

# Step 7: Verify installation
echo "📋 Step 7: Verifying installation..."

# Count tables
TABLE_COUNT=$(mysql -u root sistem_parkir_smk -e "SHOW TABLES;" 2>/dev/null | wc -l)
TABLE_COUNT=$((TABLE_COUNT - 1))  # Remove header line

if [ "$TABLE_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✅ Found $TABLE_COUNT tables${NC}"

    # Show table details
    echo ""
    echo "📊 Database contents:"
    mysql -u root sistem_parkir_smk -e "
        SELECT 'kendaraan_terdaftar' as tabel, COUNT(*) as jumlah_record FROM kendaraan_terdaftar
        UNION ALL
        SELECT 'log_akses_masuk' as tabel, COUNT(*) as jumlah_record FROM log_akses_masuk;
    "
else
    echo -e "${RED}❌ No tables found!${NC}"
    exit 1
fi

echo ""
echo "============================================================"
echo -e "${GREEN}✅ MySQL Setup Complete!${NC}"
echo "============================================================"
echo ""
echo "📝 Connection Details:"
echo "   Host: localhost"
echo "   Port: 3306"
echo "   User: root"
echo "   Password: (empty)"
echo "   Database: sistem_parkir_smk"
echo ""
echo "🚀 Next Steps:"
echo "   1. Run test: python3 test_mysql_connection.py"
echo "   2. Start app: python3 app.py"
echo "   3. Open browser: http://localhost:5001"
echo ""
echo "💡 Useful Commands:"
echo "   Stop MySQL:  brew services stop mysql"
echo "   Start MySQL: brew services start mysql"
echo "   Check status: brew services list | grep mysql"
echo "   MySQL client: mysql -u root sistem_parkir_smk"
echo ""
