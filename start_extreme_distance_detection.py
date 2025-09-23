#!/usr/bin/env python3
"""
Enhanced Motorcycle Detection untuk Jarak Jauh / Extreme Distance
Khusus untuk kondisi seperti screenshot: motor kecil, sudut tinggi, jarak jauh
"""

import argparse
import sys
import os

def main():
    """Main launcher function untuk extreme distance detection"""
    print("🔭 Extreme Distance Motorcycle Detection")
    print("Optimized untuk: Motor kecil, jarak jauh, sudut tinggi")
    print("="*55)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Extreme Distance Motorcycle Detection')
    parser.add_argument('--source', '-s', type=str, default='0',
                        help='Video source: RTSP URL, webcam index (0), or video file')
    parser.add_argument('--confidence', '-c', type=float, default=0.25,
                        help='Lower confidence for distant objects (default: 0.25)')
    parser.add_argument('--test-mode', action='store_true',
                        help='Run in test mode (60 seconds)')
    parser.add_argument('--test-duration', type=int, default=60,
                        help='Test duration in seconds (default: 60)')
    parser.add_argument('--save-crops', action='store_true',
                        help='Save detected plate crops for analysis')
    
    args = parser.parse_args()
    
    print(f"📹 Video Source: {args.source}")
    print(f"🎯 Confidence: {args.confidence} (very low for distant detection)")
    print(f"🔍 Extreme upscaling: 8x factor")
    print(f"📏 Min plate size: 15x8 pixels")
    
    if args.test_mode:
        print(f"🧪 Test Mode: {args.test_duration} seconds")
        print("="*55)
        
        # Prepare command untuk extreme distance detection
        base_cmd = f"python3 test_motorcycle_detection.py --source {args.source} --duration {args.test_duration}"
        
        print(f"🚀 Running extreme distance test...")
        print(f"⚙️ Command: {base_cmd}")
        print()
        print("💡 Tips untuk jarak jauh:")
        print("   - Pastikan kamera fokus tajam")
        print("   - Lighting yang cukup terang")
        print("   - Stabilitas kamera (tidak goyang)")
        print("   - Hindari motion blur")
        print()
        
        exit_code = os.system(base_cmd)
        
        if exit_code == 0:
            print("\n✅ Test completed successfully!")
            print("📊 Check hasil detection rate dan plate quality")
        else:
            print("\n❌ Test failed or interrupted")
            
        sys.exit(exit_code)
    else:
        print("🚀 Production Mode - Extreme Distance")
        print("="*55)
        
        # Build command untuk production
        cmd_parts = [
            "python3 main.py",
            f"--source {args.source}",
            "--motorcycle-mode",
            f"--motorcycle-confidence {args.confidence}"
        ]
        
        cmd = " ".join(cmd_parts)
        
        print(f"🚀 Running: {cmd}")
        print()
        print("💡 Extreme Distance Tips:")
        print("   - Yellow boxes = motor plates (mungkin sangat kecil)")
        print("   - Confidence akan lebih rendah untuk jarak jauh")
        print("   - Focus pada movement detection")
        print("   - Plat mungkin tidak selalu terbaca dengan akurat")
        print()
        print("⌨️ Controls:")
        print("   - 'q' to quit")
        print("   - 's' to save screenshot dengan semua detection")
        print("   - 'p' to print statistics")
        print()
        
        exit_code = os.system(cmd)
        sys.exit(exit_code)

if __name__ == "__main__":
    main()