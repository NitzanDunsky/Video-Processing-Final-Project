import cv2
import common_utils as utils
import numpy as np
from scipy.stats import gaussian_kde
from tqdm import tqdm 

MorphKernelSize = 3 # Morphological Opertations Kernel Size

def extract_certainty_regions(binary_frame): 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MorphKernelSize, MorphKernelSize))
    eroded = cv2.erode(binary_frame, kernel, iterations=1)
    dilated = cv2.dilate(binary_frame, kernel, iterations=1)
    _, uncertainty_mask = cv2.threshold(dilated - eroded, 127, 255, cv2.THRESH_BINARY)
    _, fg_certain_mask = cv2.threshold(eroded, 127, 255, cv2.THRESH_BINARY)
    _, bg_certain_mask = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY_INV)
    
    return fg_certain_mask.astype(np.uint8), \
                bg_certain_mask.astype(np.uint8), \
                    uncertainty_mask.astype(np.uint8)

def estimate_conditional_distributions(stabilized_frame, fg_certain_mask, bg_certain_mask, uncertainty_mask):
    fg = cv2.bitwise_and(stabilized_frame, stabilized_frame, mask=fg_certain_mask)
    bg = cv2.bitwise_and(stabilized_frame, stabilized_frame, mask=bg_certain_mask)
    uc = cv2.bitwise_and(stabilized_frame, stabilized_frame, mask=uncertainty_mask)

    fg_gray = cv2.cvtColor(fg, cv2.COLOR_RGB2GRAY)
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)
     
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MorphKernelSize, MorphKernelSize))
    large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*MorphKernelSize, 2*MorphKernelSize))

    erode_fg_certain_mask = np.zeros_like(fg_gray)
    erode_fg_certain_mask = cv2.erode(fg_certain_mask, kernel, iterations=1)
    more_erode_fg_certain_mask = cv2.erode(fg_certain_mask, large_kernel, iterations=1)
    fg_internal_edges = erode_fg_certain_mask-more_erode_fg_certain_mask
    
    erode_bg_certain_mask = np.zeros_like(bg_gray)
    erode_bg_certain_mask = cv2.erode(bg_certain_mask, kernel, iterations=1)
    more_erode_bg_certain_mask = cv2.erode(bg_certain_mask, large_kernel, iterations=1)
    bg_internal_edges = erode_bg_certain_mask-more_erode_bg_certain_mask

    fg_scribbles_sampled = fg[fg_internal_edges == 255]
    bg_scribbles_sampled = bg[bg_internal_edges == 255]
    fg_scribbles_sampled = fg_scribbles_sampled[::20]
    bg_scribbles_sampled = bg_scribbles_sampled[::20]


    fg_kde = gaussian_kde(fg_scribbles_sampled.T)
    bg_kde = gaussian_kde(bg_scribbles_sampled.T)

    fg_pdf = fg_kde.evaluate(uc[np.nonzero(uncertainty_mask)].T)
    bg_pdf = bg_kde.evaluate(uc[np.nonzero(uncertainty_mask)].T) 
       
    return fg_pdf, bg_pdf


def create_alpha_frame(stabilized_frame, binary_frame):    
    # gets certainty zones
    fg_certain_mask, bg_certain_mask, uncertainty_mask = extract_certainty_regions(binary_frame)

    # estimates conditional ditributions (fg, fb) over foreground and background (this is why we need the stabilized rgb frame)
    fg_pdf, bg_pdf = estimate_conditional_distributions(stabilized_frame, fg_certain_mask, bg_certain_mask, uncertainty_mask)

    # creates and return a new mask of 1 on foreground, 0 on background and f(fg, fb) on the uncertain zone 
    prob_map = fg_pdf / (fg_pdf + bg_pdf)
    alpha_frame = np.zeros(binary_frame.shape).astype(np.float32)
    alpha_frame[np.nonzero(fg_certain_mask)] = 1
    alpha_frame[np.nonzero(bg_certain_mask)] = 0
    alpha_frame[np.nonzero(uncertainty_mask)] = prob_map
    return (255 * alpha_frame).astype(np.uint8)


def create_alpha(stabilized_cap, binary_cap, output_path=f'Outputs/alpha.avi'):

    # Create an output video writers
    parameters = utils.get_video_parameters(stabilized_cap)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define video codec
    alpha_out = cv2.VideoWriter(output_path, fourcc, parameters["fps"], (parameters["width"], parameters["height"]), isColor=False)

    stabilized_frames = utils.get_frames(stabilized_cap)
    binary_frames = utils.get_frames(binary_cap)
    binary_frames = [cv2.cvtColor(binary_frame, cv2.COLOR_BGR2GRAY) for binary_frame in binary_frames]
    
    for stabilized_frame, binary_frame in tqdm(zip(stabilized_frames, binary_frames), f"{utils.now()}: AlphaMap", total=len(stabilized_frames)):
        alpha_frame = create_alpha_frame(stabilized_frame, binary_frame)
        alpha_out.write(alpha_frame)
    utils.release_videos([stabilized_cap, binary_cap, alpha_out])
    del stabilized_frames, binary_frames

    
def create_matted(extracted_cap, 
                  alpha_cap, 
                  new_bg_path='Inputs/background.jpg', 
                  output_path=utils.matted_path):
    
    # Create an output video writers
    parameters = utils.get_video_parameters(extracted_cap)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define video codec
    matted = cv2.VideoWriter(output_path, fourcc, parameters["fps"], (parameters["width"], parameters["height"]), isColor=True)
    
    # Get videos frames
    alpha_frames = np.array(utils.get_frames(alpha_cap)).astype(np.float32) / 255
    extracted_frames = utils.get_frames(extracted_cap)
    new_bg = cv2.imread(new_bg_path)
    
    # resizing background to video shape
    if new_bg.shape != (parameters["height"], parameters["width"]):
        new_bg = cv2.resize(new_bg, (parameters["width"], parameters["height"]))
    
    for alpha_frame, extracted_frame in tqdm(zip(alpha_frames, extracted_frames), f"{utils.now()}: Matting", total=len(alpha_frames)):
        matted_frame = alpha_frame * extracted_frame + (1 - alpha_frame) * new_bg
        matted.write(matted_frame.astype(np.uint8))
    # Release and close videos
    utils.release_videos([extracted_cap, alpha_cap, matted])
    del alpha_frames, extracted_frames