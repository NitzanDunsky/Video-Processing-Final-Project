import cv2
import common_utils as utils
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

def find_tracking_window(binary_image):   
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    # Find the largest contour (assuming it represents the desired object)
    largest_contour = max(contours, key=cv2.contourArea)
    # Get the bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    return (x, y, w, h)


def crop_roi(frame, window):
    x, y, w, h = window
    return frame[y : y + h + 1, x : x + w + 1]


def save_tracking_window(frame_idx, tracking_window):
    x, y, w, h = tracking_window
    utils.tracking[f"{frame_idx + 1}"] = [x, y, h, w] # start frame counting from 1 rather than 0

def track_object(matted_cap, binary_cap, output_path=utils.output_path):

    parameters = utils.get_video_parameters(matted_cap)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define video codec
    output = cv2.VideoWriter(output_path, fourcc, parameters["fps"], (parameters["width"], parameters["height"]), isColor=True)
    
    matted_frames = utils.get_frames(matted_cap)
    binary_frames = utils.get_frames(binary_cap)

    # Get rectangle (both the coordinates and cropped segment of the original frame)
    for frame_idx, (binary_frame, matted_frame) in enumerate(tqdm(zip(binary_frames, matted_frames), f"{utils.now()}: Tracking", total=len(matted_frames))):
        tracking_window = find_tracking_window(binary_frame)
        save_tracking_window(frame_idx, tracking_window)
        roi = crop_roi(matted_frame, tracking_window)
        x, y, w, h = tracking_window
        cv2.rectangle(matted_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        output.write(matted_frame)

    utils.release_videos([matted_cap, binary_cap, output])
    del binary_frames, matted_frames