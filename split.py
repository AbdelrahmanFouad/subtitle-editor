import os
import subprocess
import json

def get_duration(file_path):
    """Gets the duration of a video file in seconds using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', '-show_streams', file_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"ffprobe error: {result.stderr}")
    data = json.loads(result.stdout)
    return float(data['format']['duration'])

def split_video(file_path):
    """Splits an MP4 file into two halves if it is longer than 15 minutes."""
    duration = get_duration(file_path)
    
    # 15 minutes =  seconds
    if duration <= 1500:
        print(f"Skipping {file_path}: Duration is {duration/60:.2f} mins (under 15 mins).")
        return

    midpoint = duration / 2
    base_name = os.path.splitext(file_path)[0]
    
    part1 = f"{base_name}_part1.mp4"
    part2 = f"{base_name}_part2.mp4"

    print(f"Splitting {file_path} ({duration/60:.2f} mins) at {midpoint/60:.2f} mins...")

    # Split Part 1 (Start to Midpoint)
    subprocess.run([
        'ffmpeg', '-i', file_path, '-t', str(midpoint), 
        '-c', 'copy', '-map', '0', part1, '-y'
    ], capture_output=True)

    # Split Part 2 (Midpoint to End)
    subprocess.run([
        'ffmpeg', '-ss', str(midpoint), '-i', file_path, 
        '-c', 'copy', '-map', '0', part2, '-y'
    ], capture_output=True)

    print(f"Successfully created: \n - {part1} \n - {part2}")

if __name__ == "__main__":
    # Scan current directory for mp4 files
    files = [f for f in os.listdir('.') if f.lower().endswith('.mp4')]
    
    if not files:
        print("No MP4 files found in the current folder.")
    else:
        for f in files:
            try:
                split_video(f)
            except Exception as e:
                print(f"Error processing {f}: {e}")
