import cv2
import common_utils as utils
import numpy as np
from tqdm import tqdm 


def estimate_background(cap):
    frames = utils.get_frames(cap)
    return np.median(frames, axis=0)

def warp_frame_and_write(frame, M, out_video, estimated_background=None):
    # Create an additional channel with constant value 255 for edges masking 
    constant_channel = np.full(frame.shape[:2], 255, dtype=np.uint8)
    merged_frame = np.dstack([frame, constant_channel])
    # Apply the transformation to the current frame
    warped_frame = cv2.warpPerspective(merged_frame, M, (merged_frame.shape[1], merged_frame.shape[0]))
    #margins_mask = np.logical_not(warped_frame[:,:,3])
    margins_mask = warped_frame[:,:,3]
    warped_frame = warped_frame[..., :3]
       
    if estimated_background is not None:
        # crop the images by 5% to reduce margins, and resize to original shape
        original_height, original_width = warped_frame.shape[:2]
        crop_percentage = 0.025  
        crop_height = int(warped_frame.shape[0] * crop_percentage)
        crop_width = int(warped_frame.shape[1] * crop_percentage)
        warped_frame = warped_frame[crop_height:-crop_height, crop_width:-crop_width]
        estimated_background = estimated_background[crop_height:-crop_height, crop_width:-crop_width]
        warped_frame = cv2.resize(warped_frame, (original_width, original_height))
        estimated_background = cv2.resize(estimated_background, (original_width, original_height))
        warped_frame[margins_mask!=255] = estimated_background[margins_mask!=255]
    
    out_video.write(warped_frame)
    return warped_frame
    
def stabilize_video(cap, output_path=utils.stabilized_path, frames_skip=1, estimated_background=None):

    frames = utils.get_frames(cap)
    reduced_frames = frames[::frames_skip]
    anchor = reduced_frames[len(reduced_frames) // 2]
    anchor_gray = cv2.cvtColor(anchor, cv2.COLOR_BGR2GRAY)

    # Create an output video writer
    parameters = utils.get_video_parameters(cap)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define video codec
    out = cv2.VideoWriter(output_path, fourcc, parameters["fps"], (parameters["width"], parameters["height"]), isColor=True)

    sift, matcher = cv2.SIFT_create(), cv2.BFMatcher()
    
    # Calculate SIFT descriptors for the top interest points in the first image
    kp1, descriptors1 = sift.detectAndCompute(anchor_gray, None)

    prev_M = None

    for i, frame in enumerate(tqdm(frames, f"{utils.now()}: Stabilization")):

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if i % frames_skip == 0: # calculate new homographic transformation
            # Calculate SIFT descriptors for the top interest points in the second image
            kp2, descriptors2 = sift.detectAndCompute(current_gray, None)
            # Match descriptors between the two images
            matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

            # filter matches to best matches
            good_matches = []
            for m1, m2 in matches:
                if m1.distance < 0.5 * m2.distance:
                    good_matches.append(m1)
                
            if len(good_matches) < 10:
                print("not enough matches")

            # organize the pairs of matched points for the findHomography function
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            # find the homographic transformation between the reference image and the current image
            current_M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 3)
            
            if i > 0:
                for j in range(i - frames_skip + 1, i):
                    # calculate weight for estimating homography for the j frame
                    wj = np.abs(j - (i - frames_skip)) / frames_skip 
                    M_estimate = (1 - wj) * prev_M + wj * current_M
                    warp_frame_and_write(frames[j], M_estimate, out, estimated_background)

            warp_frame_and_write(frame, current_M, out, estimated_background)
            prev_M = current_M.copy()
        
    utils.release_videos([cap, out])
    del frames, reduced_frames