import cv2
import os

def extract_frames(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {frame_count}")
    
    # Extract first, middle, and last frames
    indices = [0, frame_count // 2, frame_count - 1]
    
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            out_path = os.path.join(output_dir, f"frame_{i}.jpg")
            cv2.imwrite(out_path, frame)
            print(f"Saved {out_path}")
        else:
            print(f"Failed to extract frame {i}")
            
    cap.release()

if __name__ == "__main__":
    extract_frames("success_run.mp4", "C:/Users/Jenson/.gemini/antigravity/brain/38d07cdd-ae6c-435d-9296-21ef258a1c50/video_debug")
