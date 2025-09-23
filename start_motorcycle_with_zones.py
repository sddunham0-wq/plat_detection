#!/usr/bin/env python3
"""
All-in-One Motorcycle Detection Launcher
Includes regular, extreme distance, and zone-based detection
"""

import argparse
import sys
import os

def main():
    """Unified launcher untuk semua mode motorcycle detection"""
    print("🏍️ Complete Motorcycle Detection System")
    print("="*50)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Complete Motorcycle Detection System')
    
    # Source and basic settings
    parser.add_argument('--source', '-s', type=str, default='0',
                        help='Video source: RTSP URL, webcam index (0), or video file')
    parser.add_argument('--confidence', '-c', type=float, default=0.4,
                        help='Detection confidence (default: 0.4)')
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--regular', action='store_true',
                           help='Regular motorcycle detection mode')
    mode_group.add_argument('--extreme', action='store_true',
                           help='Extreme distance detection mode')
    mode_group.add_argument('--zone', choices=['sequential', 'parallel', 'adaptive'],
                           help='Zone-based detection mode')
    
    # Additional options
    parser.add_argument('--test-mode', action='store_true',
                        help='Run in test mode (limited duration)')
    parser.add_argument('--test-duration', type=int, default=60,
                        help='Test duration in seconds (default: 60)')
    
    args = parser.parse_args()
    
    print(f"📹 Video Source: {args.source}")
    print(f"🎯 Confidence: {args.confidence}")
    
    # Route to appropriate detection mode
    if args.regular:
        print("🚀 Mode: Regular Motorcycle Detection")
        print("="*50)
        
        if args.test_mode:
            cmd = f"python3 test_motorcycle_detection.py --source {args.source} --duration {args.test_duration}"
        else:
            cmd = f"python3 start_motorcycle_detection.py --source {args.source} --confidence {args.confidence}"
        
        print(f"Running: {cmd}")
        print()
        print("💡 Regular Mode Features:")
        print("   ✅ Standard motorcycle detection")
        print("   ✅ Yellow bounding boxes for motorcycle plates")
        print("   ✅ Real-time OCR")
        print()
        
    elif args.extreme:
        print("🔭 Mode: Extreme Distance Detection")
        print("="*50)
        
        if args.test_mode:
            cmd = f"python3 start_extreme_distance_detection.py --test-mode --source {args.source} --confidence {args.confidence}"
        else:
            cmd = f"python3 start_extreme_distance_detection.py --source {args.source} --confidence {args.confidence}"
        
        print(f"Running: {cmd}")
        print()
        print("💡 Extreme Distance Features:")
        print("   ✅ 8x upscaling for tiny plates")
        print("   ✅ Ultra-low confidence detection")
        print("   ✅ Optimized for distant motorcycles")
        print()
        
    elif args.zone:
        print(f"🎯 Mode: Zone-Based Detection ({args.zone})")
        print("="*50)
        
        duration_arg = f"--duration {args.test_duration}" if args.test_mode else ""
        cmd = f"python3 start_zone_detection.py --source {args.source} --mode {args.zone} {duration_arg}"
        
        print(f"Running: {cmd}")
        print()
        print("💡 Zone-Based Features:")
        print(f"   ✅ 3x3 zone grid ({args.zone} mode)")
        print("   ✅ Optimized for parking lots")
        print("   ✅ Per-zone statistics")
        print("   ✅ Interactive controls (z=zones, m=mode)")
        print()
        
        if args.zone == "sequential":
            print("🎯 Sequential: Optimal untuk jarak jauh & parking lots")
        elif args.zone == "parallel":
            print("⚡ Parallel: Comprehensive detection semua zones")
        elif args.zone == "adaptive":
            print("🧠 Adaptive: Auto-adjust berdasarkan aktivitas")
        print()
    
    # Show common controls
    print("⌨️  Common Controls:")
    print("   'q' - Quit")
    print("   's' - Save screenshot")
    print("   'p' - Print statistics")
    if args.zone:
        print("   'z' - Toggle zone overlay")
        print("   'm' - Switch detection mode")
    print()
    
    # Execute command
    try:
        exit_code = os.system(cmd)
        
        if exit_code == 0:
            print("✅ Detection completed successfully!")
        else:
            print("❌ Detection failed or was interrupted")
            
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()