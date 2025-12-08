import cv2
import pandas as pd
from pathlib import Path
import numpy as np
import os

# Create directories
VIDEO_DIR = Path("data/raw_videos")
FRAMES_DIR = Path("data/frames")
SCENES_DIR = Path("data/scenes")
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)
SCENES_DIR.mkdir(exist_ok=True)


def get_video_metadata(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return fps, duration, width, height


def extract_frames(video_path, fps=2):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    frame_idx = 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    sample_rate = max(1, int(video_fps / fps))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % sample_rate == 0:
            frames.append({
                'frame_idx': frame_idx,
                'timestamp': frame_idx / video_fps,
                'frame': frame.copy()
            })
        frame_idx += 1
    cap.release()
    return frames


import numpy as np  # Add this at top if missing


def detect_scenes_fixed_interval(frames, video_path):
    """Every 8 seconds = guaranteed good scenes (Float-safe)"""
    scenes = []
    scene_interval = 8.0  # seconds
    total_duration = frames[-1]['timestamp']

    # Float-safe time steps
    start_times = np.arange(0, total_duration, scene_interval)

    for start_time in start_times:
        end_time = start_time + scene_interval

        # Find frames in window
        scene_frames = [f for f in frames if start_time <= f['timestamp'] < end_time]
        if scene_frames and len(scene_frames) > 2:
            mid_frame = scene_frames[len(scene_frames) // 2]

            # Skip black
            gray = cv2.cvtColor(mid_frame['frame'], cv2.COLOR_BGR2GRAY)
            if np.mean(gray) > 30:
                scene_path = SCENES_DIR / f"{video_path.stem}_scene_{len(scenes):02d}.jpg"
                cv2.imwrite(str(scene_path), mid_frame['frame'])

                scenes.append({
                    'scene_id': len(scenes),
                    'video_name': video_path.name,
                    'start_time': float(start_time),
                    'end_time': float(min(end_time, total_duration)),
                    'duration': float(min(end_time, total_duration) - start_time),
                    'frame_count': len(scene_frames),
                    'rep_frame_path': str(scene_path),
                    'method': 'fixed_8s'
                })

    return scenes


# === RUN PIPELINE ===
print("GACS Stage 1-2 (OpenCV Scene Detection)")
video_records = []
scene_records = []

videos_found = list(VIDEO_DIR.glob("*.mp4"))
if not videos_found:
    print("No .mp4 files in data/raw_videos/")
    print("Create folder and add your 3 IMDb videos")
    exit()

for video_file in videos_found:
    print(f"\nProcessing: {video_file.name}")

    # 1. Metadata
    fps, duration, width, height = get_video_metadata(video_file)
    video_records.append({
        'filename': video_file.name,
        'duration': round(duration, 2),
        'fps': round(fps, 2),
        'width': width,
        'height': height,
        'path': str(video_file)
    })

    # 2. Extract frames (2fps)
    frames = extract_frames(video_file)
    frame_dir = FRAMES_DIR / video_file.stem
    frame_dir.mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(frames):
        cv2.imwrite(str(frame_dir / f"frame_{i:06d}.jpg"), f['frame'])

    print(f"   {len(frames)} frames extracted")

    # 3. Scene detection
    scenes = detect_scenes_fixed_interval(frames, video_file)
    scene_records.extend(scenes)

    print(f"   {len(scenes)} scenes detected")

# Save GACS CSVs
pd.DataFrame(video_records).to_csv(DATA_DIR / "video_metadata.csv", index=False)
pd.DataFrame(scene_records).to_csv(DATA_DIR / "scene_metadata.csv", index=False)

print(f"\nGACS Stage 1-2 COMPLETE!")
print(f"{len(video_records)} videos → {len(scene_records)} scenes")
print("data/video_metadata.csv")
print("data/scene_metadata.csv")
print("Ready for Gemini labeling!")
