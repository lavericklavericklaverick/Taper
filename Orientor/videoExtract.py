import cv2
import os
import math
from tqdm import tqdm

def extract_frames_from_video(video_path, output_dir="raw_images", frames_per_second=3):
    """
    Extracts frames from a video file at a specified rate (default: 3 frames per second)
    and saves them as PNG images for labeling.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return

    # Get video frame rate (e.g., 30 FPS)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = 30  # Fallback standard
        
    sample_interval = max(1, int(video_fps / frames_per_second))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_samples = math.ceil(total_frames / sample_interval) if total_frames > 0 else None

    saved_count = 0
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    with tqdm(total=total_samples, desc="Extracting frames", unit="frame") as progress:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_filename = f"{base_name}_frame_{saved_count:04d}.png"
            save_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(save_path, frame)
            saved_count += 1
            progress.update(1)

            for _ in range(sample_interval - 1):
                if not cap.grab():
                    break

            if not cap.isOpened():
                break

    cap.release()
    print(f"Extracted {saved_count} frames from '{video_path}' into '{output_dir}'.")

if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select a video",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()

    if video_path:
        extract_frames_from_video(video_path)
    else:
        print("No video selected.")