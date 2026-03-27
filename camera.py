##############################################################
#
#  Interacts with an Intel RealSense D405 Camera
#   
#  Code Created by Benjamin Gorse for Princeton 2024 Sr Thesis
#
##############################################################

import cv2
import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

# Camera Specs:
DEPTH_H_RES = 1280                                  # pixels
DEPTH_V_RES = 720                                   # pixels
DEPTH_FRAME_RATE = 30                               # FPS
COLOR_H_RES = 1280                                  # pixels
COLOR_V_RES = 720                                   # pixels
COLOR_FRAME_RATE = 90                               # FPS
MATCH_FR = min(COLOR_FRAME_RATE, DEPTH_FRAME_RATE)  # FPS

class IntelD405:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        depth_sensor = device.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale() # In meters
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        # Creates depth stream in 16 bit format--each pixel stores depth distance with little endian order
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Used for aligning depth and color frames and getting the color map of depth
        self.colorizer = rs.colorizer()
        self.align = rs.align(rs.stream.color)

        # Start streaming
        self.pipeline.start(config)

        # Set up the stuff for realsense post-processing:
        self.spatial = rs.spatial_filter()
        self.decimation = rs.decimation_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)

        # Ignore the first 5 frames to give the Auto-Exposure time to adjust
        for x in range(5):
            self.pipeline.wait_for_frames()


    def get_frames(self, as_array=True, keep=False):
        """
        Collects a color and depth frame from the Camera:
        Returns ret
        ret[0] : bool
            True if able to return a frame, False for an error
        ret[1] : np array
            if ret[0] == True, is array for depth image
        ret[2] : np array
            if ret[0] == True, is array for color image
        ret[3] : np array
            if ret[0] == True, is array for the raw depth data
        """
        # Get frame from stream
        frames = self.pipeline.wait_for_frames()

        # This is an initial realsense post process which should be done before the alignment occurs
        frames = self.rs_post_process(frames)

        # aligning color and depth frames to make directly comparable
        aligned_frames = self.align.process(frames)          
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()

        if keep == True:
            aligned_frames.keep()

        # If one was not received, return False and no frames
        if aligned_depth_frame == None: 
            return False, None, None
        if aligned_color_frame == None:
            return False, None, None

        # converts depth/color frames to np arrays
        if as_array:
            aligned_depth_frame = np.asanyarray(aligned_depth_frame.get_data())
            aligned_color_frame = np.asanyarray(aligned_color_frame.get_data())

        # On success, return True and both frames
        return True, aligned_depth_frame, aligned_color_frame
    
    def get_frame(self, type, as_array=True):
        """
        Collects either a color or depth frame from the Camera:
        Returns
        [0] : bool
            True if able to return a frame, False for an error
        [1] : np array
            image which matches type specified
        """
        # Get frame from stream
        if type.lower() != "color" and type.lower() != "depth":
            raise RuntimeError("`type` parameter must be 'color' or 'depth'")
        frames = self.pipeline.wait_for_frames()
        # If one was not received, return False and no frames
        
        if type.lower() == "color":
            color_frame = frames.get_color_frame()
            if color_frame == None:
                return False, None, None
            if as_array:
                color_frame = np.asanyarray(color_frame.get_data())
            return True, color_frame
        else:
            depth_frame = frames.get_depth_frame()
            if depth_frame == None:
                return False, None, None
            frames = self.align.process(frames)

            if as_array:
                depth_frame = np.asanyarray(depth_frame.get_data())
            # On success, return True and both frames
            return True, depth_frame
    
    def show_raws(self):
        """
        Shows the aligned raw feeds using CV2. 
        Returns:
        * True if able to show frames
        * False if no frames to show
        """
        success, depth, color = self.get_frames(as_array=False)
        
        if success:
            colorized_depth = np.asanyarray(self.colorizer.colorize(depth).get_data())
            cv2.imshow("depth frame", colorized_depth)
            color = np.asanyarray(color.get_data())
            cv2.imshow("Color frame", color)
            return True
        return False

    def release(self):
        """
        Ends the Camera pipeline
        """
        self.pipeline.stop()

    def rs_post_process(self, frames):
        """
        This is to provide an initial post processing of the image in order to remove defects. 
        This is just a basic pass to correct color and depth defects. 

        Parameters:
            frames : Returned object from rs.pipeline.wait_for_frame()
        """

        # SPATIAL FILTER: smooths depth frame while maintaining lines
        # Number of filter iterations
        self.spatial.set_option(rs.option.filter_magnitude, 2)         # Starting default: 2, [1-5]
        # The alpha factor for an exponential moving average 
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)    # Starting default: 0.6, [0.25-1] where 0.25 is most aggressive
        # The following establishes the thresold value for edges. Higher threshold, the harder it will be to see an edge
        self.spatial.set_option(rs.option.filter_smooth_delta, 10)      # Starting default: 20, [1-50] 
        # The following decides the maximum size for holes (gaps in depth calculation) that can be filled in
        self.spatial.set_option(rs.option.holes_fill, 5)                # Starting default: 3, [0-5] (0 being none, 5 being all, 3=8 pixels)

        # HOLE FILLING: Fills the holes that are inherent to depth sensor when info is not precise:
        self.hole_filling.set_option(rs.option.holes_fill, 1)

        # Execute the filters (needs to be in this particular order)
        frames = self.depth_to_disparity.process(frames)
        frames = self.spatial.process(frames)
        frames = self.disparity_to_depth.process(frames).as_frameset()

        depth_frame = frames.get_depth_frame()
        depth_frame = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())

        cv2.imshow("post process", depth_frame)

        return frames


    def record_video(self, file_name, fps, duration):
        """
        Records a video from the camera and stores the result in the file called
        `file_name`+"_depth" and `file_name`+"_color". If `duration` is not provided, 
        then it records a video until the button 'q' is pressed. Video records 
        for `duration` seconds if specified. 
        """
        color_file_name = file_name + "_color.avi"
        depth_file_name = file_name + "_depth.avi"
        depth_data_file_name = file_name + "_depth.pkl" 
        depth_frames = []   # stores the individual frame data for the depth cam to be stored in .npy file

        success, depth_frame, color_frame = self.get_frames(as_array=True) 
        
        if not(success):
            return False
        color_image_size = (color_frame.shape[1], color_frame.shape[0])
        depth_image_size = (depth_frame.shape[1], depth_frame.shape[0])
        color_video = cv2.VideoWriter(color_file_name,cv2.VideoWriter_fourcc('M','J','P','G'), fps, color_image_size, True)
        depth_video = cv2.VideoWriter(depth_file_name,cv2.VideoWriter_fourcc('M','J','P','G'), fps, depth_image_size, True)

        print("Video Recording Starting in:\n3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)
        print("START")

        if duration:
            target_length = fps*duration
            counter = 0
            while True:  
                success, depth, color = self.get_frames(as_array=False, keep=True)
                print("Successful get frame #{}".format(counter+1))
        
                if success:
                    colorized_depth = np.asanyarray(self.colorizer.colorize(depth).get_data())
                    depth = np.asanyarray(depth.get_data())
                    cv2.imshow("depth frame", colorized_depth)
                    color = np.asanyarray(color.get_data())
                    cv2.imshow("Color frame", color)
                    depth_video.write(colorized_depth)
                    color_video.write(color)
                    counter += 1
                    depth_frames.append(depth)
                else:
                    print("FAILED FRAME")

                if counter == target_length:
                    break
        else:
            while True:
                success, depth, color = self.get_frames(as_array=False, keep=True)
        
                if success:
                    colorized_depth = np.asanyarray(self.colorizer.colorize(depth).get_data())
                    depth = np.asanyarray(depth.get_data())
                    
                    cv2.imshow("depth frame", colorized_depth)
                    color = np.asanyarray(color.get_data())
                    cv2.imshow("Color frame", color)

                    depth_video.write(colorized_depth)
                    color_video.write(color)
                    depth_frames.append(depth)
                else:
                    print("FAILED FRAME")

                # Hit q to quit.
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        print("RECORDING COMPLETE")
        color_video.release()
        depth_video.release()

        # Saves collection of depth frames to a files to be accessed later
        with open(depth_data_file_name, "wb") as depth_file:   
            pickle.dump(depth_frames, depth_file)


def main():
    """
    Runs testing suite to verify function of the IntelD405 class
    """

    cam = IntelD405()

    while (True):
        cam.show_raws()

        # Hit q to quit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    file_name = "incamera_test"
    fps = 30
    duration = 5
    cam.record_video(file_name=file_name, fps=fps, duration=duration)
    
    cam.release()


# Runs the main method upon execution of the file
if __name__ == '__main__': 
    main()