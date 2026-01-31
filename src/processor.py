import cv2
import os

def extract_frames(video_path, output_folder, interval=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success, image = vidcap.read()
    count = 0
    saved_count = 0

    while success:
        if count % (int(fps) * interval) == 0:
            frame_name = os.path.join(output_folder, f"frame_{saved_count}.jpg")
            cv2.imwrite(frame_name, image)
            saved_count += 1
        success, image = vidcap.read()
        count += 1
    
    vidcap.release()
    return saved_count