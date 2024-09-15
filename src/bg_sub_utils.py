import cv2
import common_utils as utils
import numpy as np
from datetime import datetime
from tqdm import tqdm 
import random

MedFilterSize = 11 # Median Filter Kernel Size
MorphKernelSize = 5 # Morphological Opertations Kernel Size
MorphIterations = 3 # Morphological Operations Number of Iterations
PreLearningIterations = 3

def subtract_background(cap, 
                        output_path_binary=utils.binary_path, 
                        output_path_extracted=utils.extracted_path):

    # Create an output video writers
    parameters = utils.get_video_parameters(cap)
    frames = utils.get_frames(cap)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define video codec
    binary_out = cv2.VideoWriter(output_path_binary, fourcc, parameters["fps"], (parameters["width"], parameters["height"]), isColor=False)
    extracted_out = cv2.VideoWriter(output_path_extracted, fourcc, parameters["fps"], (parameters["width"], parameters["height"]), isColor=True)

    bg_subtractor = cv2.createBackgroundSubtractorKNN()

    # Pre-Learning Phase
    training_frames = frames.copy()
    training_frames.reverse()
    training_frames.extend(frames)
    training_frames.reverse()
    for _ in tqdm(range(PreLearningIterations), f"{utils.now()}: BackgroundSubtraction | KNN Training"):
        for frame in training_frames:
            bg_subtractor.apply(frame)
    del training_frames

    for i, frame in enumerate(tqdm(frames, f"{utils.now()}: BackgroundSubtraction | Execution")):

        # Pre-Processing    
        blurred = cv2.GaussianBlur(frame, ksize=(5,5), sigmaX=2)

        # Predict
        fg_mask = bg_subtractor.apply(blurred)

        # Post-processing
        fg_mask = cv2.medianBlur(fg_mask, MedFilterSize)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MorphKernelSize, MorphKernelSize))
        
        # # Opening operation for noise reduction
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=MorphIterations)

        # # Closing for filling the moving character
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=MorphIterations)

        # Apply threshold to obtain a binary mask
        _, fg_mask = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)
              
        # Extract the desired content from the original frame
        extracted_frame = cv2.bitwise_and(frame, frame, mask=fg_mask)

        # Write the binary mask and the extracted frame to the output videos
        binary_out.write(fg_mask)
        extracted_out.write(extracted_frame)
    
    utils.release_videos([cap, binary_out, extracted_out])
    del frames