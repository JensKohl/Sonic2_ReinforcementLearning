import cv2
import sys
import os

def debug_video(video_path):
    print(f"Checking video: {video_path}")
    if not os.path.exists(video_path):
        print("File does not exist.")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Video did not open.")
        return
    
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Properties: {width}x{height}, {fps} fps, {count} frames")
    
    # Save the very last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, count - 1)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("last_frame_debug.jpg", frame)
        print("Saved last_frame_debug.jpg")
    else:
        # Try reading frame by frame till the end
        print("Failed to seek to end. Reading sequentially...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        last_success = None
        for i in range(count):
            ret, frame = cap.read()
            if ret:
                last_success = frame
            else:
                break
        if last_success is not None:
             cv2.imwrite("last_frame_debug.jpg", last_success)
             print(f"Saved last_frame_debug.jpg (last successful read at {i})")

    cap.release()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_video_2.py <video_path>")
    else:
        debug_video(sys.argv[1])
