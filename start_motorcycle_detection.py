#!/usr/bin/env python3
"""
Launcher untuk enhanced motorcycle plate detection
"""

import argparse
import sys
import os

def main():
    """Main launcher function"""
    print("🏍️ Enhanced Motorcycle Plate Detection Launcher")
    print("="*50)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Enhanced Motorcycle Plate Detection System')
    parser.add_argument('--source', '-s', type=str, default='0',
                        help='Video source: RTSP URL, webcam index (0), or video file')
    parser.add_argument('--confidence', '-c', type=float, default=0.4,
                        help='Motorcycle detection confidence (default: 0.4)')
    parser.add_argument('--test-mode', action='store_true',
                        help='Run in test mode (30 seconds)')
    parser.add_argument('--test-duration', type=int, default=30,
                        help='Test duration in seconds (default: 30)')
    
    args = parser.parse_args()
    
    print(f"📹 Video Source: {args.source}")
    print(f"🎯 Confidence: {args.confidence}")
    
    if args.test_mode:
        print(f"🧪 Test Mode: {args.test_duration} seconds")
        print("="*50)
        
        # Run test
        cmd = f"python test_motorcycle_detection.py --source {args.source} --duration {args.test_duration}"
        print(f"🚀 Running: {cmd}")
        print()
        
        exit_code = os.system(cmd)
        sys.exit(exit_code)
    else:
        print("🚀 Production Mode")
        print("="*50)
        
        # Run main system with motorcycle mode
        cmd = f"python main.py --source {args.source} --motorcycle-mode --motorcycle-confidence {args.confidence}"
        print(f"🚀 Running: {cmd}")
        print()
        print("💡 Usage Tips:")
        print("   - 'q' to quit")
        print("   - 's' to save screenshot")
        print("   - Yellow boxes = motorcycle plates")
        print("   - Green boxes = regular plates")
        print()
        
        exit_code = os.system(cmd)
        sys.exit(exit_code)

if __name__ == "__main__":
    main()