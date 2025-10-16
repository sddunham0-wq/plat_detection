#!/bin/bash
# =====================================================
# SETUP REMOTE MYSQL ACCESS - macOS Server
# =====================================================
# Script untuk enable remote access MySQL dari Windows ke macOS
#
# Usage:
#   chmod +x setup_remote_mysql.sh
#   ./setup_remote_mysql.sh

set -e  # Exit on error

echo "=========================================="
echo "  🚀 Setup Remote MySQL Access"
echo "  macOS Server Configuration"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# =====================================================
# STEP 1: Detect IP Address
# =====================================================

echo "📍 STEP 1: Detecting macOS IP Address..."
echo ""

# Get IP address
IP_ADDR=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "")

if [ -z "$IP_ADDR" ]; then
    echo -e "${RED}❌ Cannot detect IP address!${NC}"
    echo "Please run manually: ifconfig | grep 'inet '"
    exit 1
fi

echo -e "${GREEN}✅ IP Address detected: $IP_ADDR${NC}"
echo ""

# =====================================================
# STEP 2: Check MySQL Status
# =====================================================

echo "📍 STEP 2: Checking MySQL Status..."
echo ""

# Check if MySQL is running
if brew services list | grep -q "mysql.*started"; then
    echo -e "${GREEN}✅ MySQL is running${NC}"
elif pgrep -x mysqld > /dev/null; then
    echo -e "${GREEN}✅ MySQL is running${NC}"
else
    echo -e "${YELLOW}⚠️  MySQL is not running!${NC}"
    echo ""
    read -p "Do you want to start MySQL now? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Starting MySQL..."
        brew services start mysql || {
            echo -e "${RED}❌ Failed to start MySQL via Homebrew${NC}"
            echo "Try manually: sudo /usr/local/mysql/support-files/mysql.server start"
            exit 1
        }
        echo -e "${GREEN}✅ MySQL started${NC}"
        sleep 3
    else
        echo -e "${RED}❌ Cannot proceed without MySQL running${NC}"
        exit 1
    fi
fi

echo ""

# =====================================================
# STEP 3: Get MySQL Root Password
# =====================================================

echo "📍 STEP 3: MySQL Authentication..."
echo ""

read -sp "Enter MySQL root password (press Enter if empty): " MYSQL_ROOT_PASSWORD
echo ""
echo ""

# Test MySQL connection
if [ -z "$MYSQL_ROOT_PASSWORD" ]; then
    mysql -u root -e "SELECT 1" > /dev/null 2>&1 || {
        echo -e "${RED}❌ Cannot connect to MySQL as root${NC}"
        echo "Please check your MySQL root password"
        exit 1
    }
else
    mysql -u root -p"$MYSQL_ROOT_PASSWORD" -e "SELECT 1" > /dev/null 2>&1 || {
        echo -e "${RED}❌ Cannot connect to MySQL as root${NC}"
        echo "Password might be incorrect"
        exit 1
    }
fi

echo -e "${GREEN}✅ MySQL authentication successful${NC}"
echo ""

# =====================================================
# STEP 4: Check Database Exists
# =====================================================

echo "📍 STEP 4: Checking Database..."
echo ""

DB_NAME="sistem_parkir_smk"

if [ -z "$MYSQL_ROOT_PASSWORD" ]; then
    DB_EXISTS=$(mysql -u root -sse "SELECT COUNT(*) FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME='$DB_NAME'")
else
    DB_EXISTS=$(mysql -u root -p"$MYSQL_ROOT_PASSWORD" -sse "SELECT COUNT(*) FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME='$DB_NAME'")
fi

if [ "$DB_EXISTS" -eq 1 ]; then
    echo -e "${GREEN}✅ Database '$DB_NAME' exists${NC}"
else
    echo -e "${YELLOW}⚠️  Database '$DB_NAME' does not exist!${NC}"
    echo ""
    read -p "Do you want to create it from database_setup.sql? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ -f "database_setup.sql" ]; then
            echo "Creating database from database_setup.sql..."
            if [ -z "$MYSQL_ROOT_PASSWORD" ]; then
                mysql -u root < database_setup.sql
            else
                mysql -u root -p"$MYSQL_ROOT_PASSWORD" < database_setup.sql
            fi
            echo -e "${GREEN}✅ Database created${NC}"
        else
            echo -e "${RED}❌ database_setup.sql not found!${NC}"
            exit 1
        fi
    else
        echo -e "${RED}❌ Cannot proceed without database${NC}"
        exit 1
    fi
fi

echo ""

# =====================================================
# STEP 5: Create Remote User
# =====================================================

echo "📍 STEP 5: Creating Remote User..."
echo ""

# Get username and password
read -p "Enter remote username (default: remote_user): " REMOTE_USER
REMOTE_USER=${REMOTE_USER:-remote_user}

read -sp "Enter remote password (default: password_kuat_123): " REMOTE_PASSWORD
echo ""
REMOTE_PASSWORD=${REMOTE_PASSWORD:-password_kuat_123}

read -p "Allow from all IPs? (y=all, n=specific IP): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    REMOTE_HOST="%"
    echo "Will allow access from ALL IPs"
else
    read -p "Enter client IP address (Windows IP): " REMOTE_HOST
    echo "Will allow access from: $REMOTE_HOST"
fi

echo ""

# Create user SQL
CREATE_USER_SQL="
DROP USER IF EXISTS '$REMOTE_USER'@'$REMOTE_HOST';
CREATE USER '$REMOTE_USER'@'$REMOTE_HOST' IDENTIFIED BY '$REMOTE_PASSWORD';
GRANT ALL PRIVILEGES ON $DB_NAME.* TO '$REMOTE_USER'@'$REMOTE_HOST';
FLUSH PRIVILEGES;
SELECT User, Host FROM mysql.user WHERE User='$REMOTE_USER';
"

# Execute SQL
if [ -z "$MYSQL_ROOT_PASSWORD" ]; then
    echo "$CREATE_USER_SQL" | mysql -u root
else
    echo "$CREATE_USER_SQL" | mysql -u root -p"$MYSQL_ROOT_PASSWORD"
fi

echo -e "${GREEN}✅ User '$REMOTE_USER'@'$REMOTE_HOST' created${NC}"
echo ""

# =====================================================
# STEP 6: Configure bind-address
# =====================================================

echo "📍 STEP 6: Configuring MySQL bind-address..."
echo ""

# Find my.cnf location
MY_CNF_LOCATIONS=(
    "/usr/local/etc/my.cnf"
    "/opt/homebrew/etc/my.cnf"
    "/etc/my.cnf"
    "/etc/mysql/my.cnf"
)

MY_CNF=""
for location in "${MY_CNF_LOCATIONS[@]}"; do
    if [ -f "$location" ]; then
        MY_CNF="$location"
        break
    fi
done

if [ -z "$MY_CNF" ]; then
    echo -e "${YELLOW}⚠️  my.cnf not found in common locations${NC}"
    echo "You may need to create it manually at: /usr/local/etc/my.cnf"
    echo ""
    echo "Add this to [mysqld] section:"
    echo "  bind-address = 0.0.0.0"
    echo ""
    read -p "Press Enter to continue..."
else
    echo "Found my.cnf at: $MY_CNF"

    # Check if bind-address is already set
    if grep -q "^bind-address" "$MY_CNF"; then
        CURRENT_BIND=$(grep "^bind-address" "$MY_CNF" | awk '{print $3}')
        echo "Current bind-address: $CURRENT_BIND"

        if [ "$CURRENT_BIND" = "0.0.0.0" ] || [ "$CURRENT_BIND" = "$IP_ADDR" ]; then
            echo -e "${GREEN}✅ bind-address already configured correctly${NC}"
        else
            echo -e "${YELLOW}⚠️  bind-address needs to be changed${NC}"
            echo ""
            read -p "Change bind-address to 0.0.0.0? (y/n): " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                sudo sed -i.bak "s/^bind-address.*/bind-address = 0.0.0.0/" "$MY_CNF"
                echo -e "${GREEN}✅ bind-address changed to 0.0.0.0${NC}"
                NEED_RESTART=true
            fi
        fi
    else
        echo -e "${YELLOW}⚠️  bind-address not found in my.cnf${NC}"
        echo ""
        read -p "Add bind-address = 0.0.0.0 to my.cnf? (y/n): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sudo bash -c "echo 'bind-address = 0.0.0.0' >> $MY_CNF"
            echo -e "${GREEN}✅ bind-address added${NC}"
            NEED_RESTART=true
        fi
    fi
fi

echo ""

# =====================================================
# STEP 7: Restart MySQL
# =====================================================

if [ "$NEED_RESTART" = true ]; then
    echo "📍 STEP 7: Restarting MySQL..."
    echo ""

    read -p "Restart MySQL now? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Restarting MySQL..."
        brew services restart mysql || {
            echo "Homebrew restart failed, trying manual restart..."
            sudo /usr/local/mysql/support-files/mysql.server restart
        }
        echo -e "${GREEN}✅ MySQL restarted${NC}"
        sleep 3
    else
        echo -e "${YELLOW}⚠️  Remember to restart MySQL manually!${NC}"
        echo "   brew services restart mysql"
    fi
else
    echo "📍 STEP 7: MySQL Restart..."
    echo -e "${GREEN}✅ No restart needed${NC}"
fi

echo ""

# =====================================================
# STEP 8: Check Firewall
# =====================================================

echo "📍 STEP 8: Checking Firewall..."
echo ""

FIREWALL_STATUS=$(sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate | grep "enabled" || echo "disabled")

if [[ $FIREWALL_STATUS == *"enabled"* ]]; then
    echo -e "${YELLOW}⚠️  Firewall is ENABLED${NC}"
    echo "You may need to allow incoming connections on port 3306"
    echo ""
    echo "Options:"
    echo "1. Disable firewall temporarily (not recommended)"
    echo "2. Add MySQL to firewall exceptions"
    echo "3. Continue without changes (if already configured)"
    echo ""
    read -p "Choose option (1/2/3): " -n 1 -r
    echo ""

    if [[ $REPLY == "1" ]]; then
        sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate off
        echo -e "${GREEN}✅ Firewall disabled${NC}"
    elif [[ $REPLY == "2" ]]; then
        echo "Please add MySQL manually in System Settings → Network → Firewall"
        read -p "Press Enter after configuring..."
    fi
else
    echo -e "${GREEN}✅ Firewall is disabled${NC}"
fi

echo ""

# =====================================================
# SUMMARY
# =====================================================

echo "=========================================="
echo "  ✅ SETUP COMPLETED!"
echo "=========================================="
echo ""
echo "📝 Connection Details:"
echo ""
echo "  macOS Server IP: $IP_ADDR"
echo "  MySQL Port: 3306"
echo "  Database: $DB_NAME"
echo "  Username: $REMOTE_USER"
echo "  Password: $REMOTE_PASSWORD"
echo ""
echo "=========================================="
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Copy 'view_database.py' to your Windows computer"
echo ""
echo "2. Edit view_database.py and change:"
echo "   MACOS_SERVER_IP = \"$IP_ADDR\""
echo "   DB_CONFIG = {"
echo "       'user': '$REMOTE_USER',"
echo "       'password': '$REMOTE_PASSWORD',"
echo "   }"
echo ""
echo "3. On Windows, install Python library:"
echo "   pip install mysql-connector-python"
echo ""
echo "4. Run on Windows:"
echo "   python view_database.py"
echo ""
echo "=========================================="
echo ""
echo "🔧 Test Connection from Windows:"
echo ""
echo "   # Ping test"
echo "   ping $IP_ADDR"
echo ""
echo "   # Port test (PowerShell)"
echo "   Test-NetConnection -ComputerName $IP_ADDR -Port 3306"
echo ""
echo "   # MySQL test"
echo "   mysql -h $IP_ADDR -u $REMOTE_USER -p$REMOTE_PASSWORD $DB_NAME"
echo ""
echo "=========================================="
echo ""
echo -e "${GREEN}Setup completed successfully! 🎉${NC}"
echo ""
