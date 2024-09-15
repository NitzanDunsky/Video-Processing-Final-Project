import time
import json
import common_utils as utils
import stabilization_utils as stabilization
import bg_sub_utils as bg_sub
import matting_utils as matting
import time
import tracking_utils as tracking

def main() -> None:
        total_tic = time.time()

        # Stabilization
        # ------------------------------------------------------------------------------------------------------------------------
        tic = time.time()
        stabilization.stabilize_video(utils.get_video('Inputs/INPUT.avi'), frames_skip=3)
        estimated_background = stabilization.estimate_background(utils.get_video(utils.stabilized_path))
        stabilization.stabilize_video(utils.get_video('Inputs/INPUT.avi'), frames_skip=1, estimated_background=estimated_background)
        toc = time.time()
        utils.timing["time_to_stabilize"] = int(toc - tic)
        # ------------------------------------------------------------------------------------------------------------------------

        # Background Subtraction
        # ------------------------------------------------------------------------------------------------------------------------
        tic = time.time()
        bg_sub.subtract_background(utils.get_video(utils.stabilized_path))
        toc = time.time()
        utils.timing["time_to_binary"] = int(toc - tic)
        # ------------------------------------------------------------------------------------------------------------------------

        # Alpha Map
        # ------------------------------------------------------------------------------------------------------------------------
        tic = time.time()
        matting.create_alpha(utils.get_video(utils.stabilized_path), utils.get_video(utils.binary_path))
        toc = time.time()
        utils.timing["time_to_alpha"] = int(toc - tic)
        # ------------------------------------------------------------------------------------------------------------------------

        # Matting
        # ------------------------------------------------------------------------------------------------------------------------
        tic = time.time()
        matting.create_matted(utils.get_video(utils.extracted_path), utils.get_video(utils.alpha_path))
        toc = time.time()
        utils.timing["time_to_matted"] = int(toc - tic)
        # ------------------------------------------------------------------------------------------------------------------------

        # Object Tracking
        # ------------------------------------------------------------------------------------------------------------------------
        tic = time.time()
        tracking.track_object(utils.get_video(utils.matted_path), utils.get_video(utils.binary_path))
        toc = time.time()
        utils.timing["time_to_output"] = int(toc - tic)
        # ------------------------------------------------------------------------------------------------------------------------

        with open(utils.timing_path, 'w') as f:
                json.dump(utils.timing, f)

        with open(utils.tracking_path, 'w') as f:
                json.dump(utils.tracking, f)

        total_toc = time.time()
        print(f'Finished the pipeline end-to-end in {total_toc - total_tic}!')

if __name__ == "__main__":
    main()