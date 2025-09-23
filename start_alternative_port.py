#!/usr/bin/env python3
"""
Start plate detection server on alternative port
Solusi untuk masalah port 5000 conflict dengan AirPlay
"""

import sys
import os

def main():
    """Start server with alternative port"""
    print("🚀 Starting Plate Detection Server on Alternative Port")
    print("=" * 50)
    
    # Cek port yang tersedia
    available_ports = [5001, 5002, 5003, 8000, 8080, 3000]
    
    for port in available_ports:
        try:
            # Import setelah path setup
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            
            # Check if port is available
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            
            if result != 0:  # Port is available
                print(f"✅ Port {port} tersedia")
                print(f"🌐 Starting server on http://localhost:{port}")
                
                # Import dan jalankan headless_stream
                from headless_stream import main as headless_main
                headless_main(host='0.0.0.0', port=port, debug=False, no_yolo=False)
                break
            else:
                print(f"⚠️ Port {port} sedang digunakan")
                
        except Exception as e:
            print(f"❌ Error checking port {port}: {e}")
            continue
    else:
        print("❌ Tidak ada port yang tersedia dari list")
        print("💡 Coba matikan AirPlay Receiver atau gunakan:")
        print("   python3 headless_stream.py --port 9000")

if __name__ == "__main__":
    main()