import cv2
import numpy as np
from tqdm import tqdm
from scipy import signal
from datetime import datetime
import os
import json


stabilized_path = os.path.join('Outputs', f"stabilized.avi")
binary_path     = os.path.join('Outputs', f"binary.avi")
extracted_path  = os.path.join('Outputs', f"extracted.avi")
alpha_path      = os.path.join('Outputs', f"alpha.avi")
matted_path     = os.path.join('Outputs', f"matted.avi")
output_path     = os.path.join('Outputs', f"OUTPUT.avi")

timing_path = os.path.join('Outputs', f"timing.json")
timing = {"time_to_stabilize":   0,
           "time_to_binary":    0, 
           "time_to_alpha":     0, 
           "time_to_matted":    0, 
           "time_to_output":    0}

tracking_path = os.path.join('Outputs', f"tracking.json")
tracking = dict()

LPF = 1.0 / 256 * np.array([[1, 4, 6, 4, 1],
                            [4, 16, 24, 16, 4],
                            [6, 24, 36, 24, 6],
                            [4, 16, 24, 16, 4],
                            [1, 4, 6, 4, 1]])

def now():
    return datetime.now().time().replace(microsecond=0)

def get_video_parameters(capture: cv2.VideoCapture) -> dict:

    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"fourcc": fourcc, 
            "fps": fps, 
            "height": height, 
            "width": width,
            "frame_count": frame_count}
    
def get_video(path):
    cap = cv2.VideoCapture(path)
    assert(cap.isOpened()), f"Error: Video file {path} is not open!"
    return cap

def get_frames(video):
    params = get_video_parameters(video)
    num_of_frames = params["frame_count"]
    frames = []
    for i in range(num_of_frames):
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return frames

def release_videos(videos: list):
    for video in videos:
        video.release()
    cv2.destroyAllWindows()

def decimate_video(video_cap, spatial_rate, temporal_rate, output_name= 'low_res_video.avi'):
    parameters = get_video_parameters (video_cap)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_name, fourcc, parameters["fps"] // temporal_rate, (parameters["width"] // spatial_rate, parameters["height"] // spatial_rate), isColor=True)
    
    for i in tqdm(range(parameters["frame_count"]-1)):
        ret, frame = video_cap.read()
        if i % temporal_rate == 0:
            out_frame = np.zeros_like(frame)
            for ch in range(3):
                out_frame[..., ch] = signal.convolve2d(frame[..., ch], LPF, boundary='symm', mode='same')
            out_frame = out_frame[::spatial_rate, ::spatial_rate, ...]
            out.write(out_frame)
    
    return out

def build_pyramid(image: np.ndarray, num_levels: int):

    PYRAMID_FILTER = 1.0 / 256 * np.array([[1, 4, 6, 4, 1],
                                       [4, 16, 24, 16, 4],
                                       [6, 24, 36, 24, 6],
                                       [4, 16, 24, 16, 4],
                                       [1, 4, 6, 4, 1]])
    pyramid = [image.copy()]
    for l in range(num_levels):
        prev_level_img = pyramid[-1]
        next_level_img = np.zeros_like(prev_level_img)
        for ch in range(3):
            next_level_img[... ,ch] = signal.convolve2d(prev_level_img[... ,ch], PYRAMID_FILTER, boundary='symm', mode='same')
        next_level_img = next_level_img[::2, ::2]
        pyramid.append(next_level_img.astype(np.uint8))
    pyramid.reverse()
    return pyramid