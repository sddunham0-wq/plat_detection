import cv2

print("Testing webcam...")

for i in range(3):
    print(f"\nTrying index {i}...")
    cap = cv2.VideoCapture(i)
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ Webcam index {i} WORKS!")
            print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print(f"❌ Index {i} opened but can't read frame")
    else:
        print(f"❌ Index {i} not available")
    
    cap.release()

print("\nTest selesai!")