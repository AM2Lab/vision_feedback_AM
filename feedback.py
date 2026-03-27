##############################################################
#
#  Performs real time feedback control 
#  calls:  IntelD405() from camera.py
#  calls:  ImageModule() from image_module.py 
#  Code Created by Benjamin Gorse for Princeton 2024 Sr Thesis
#
##############################################################


from camera import IntelD405
from image_module import ImageModule
import time
import numpy as np
import cv2
import pickle
from duetwebapi import DuetWebAPI as DWA
import matplotlib.pyplot as plt
 

FROM_VIDEO = False
RECORD_VIDEO = False
READ_VIDEO_NAME = 'video_outputs\\16mm_fil_follow'
REC_VIDEO_NAME = "NO_OVERWRITE"

TARGET_LAYER_WIDTH = 12.0           # mm
TARGET_LAYER_DEPTH = 8.0            # mm
TARGET_HEIGHT_ABOVE_FIL = 77        # mm
TU_W = 60                           # period of oscillations in seconds
FEEDBACK_KP_W = (-1.0 / (1.5*TARGET_LAYER_WIDTH)) * 0.5
FEEDBACK_KI_W = -0.0
FEEDBACK_KD_W = (-1.0 / (3.0*TARGET_LAYER_WIDTH)) * TU_W * 0.125 
FEEDBACK_KP_D = -0.95
FEEDBACK_KI_D = -0.0
FEEDBACK_KD_D = -0.0

U_W_HI_LIM = 1.2                    # Upper limit for filter to width control response
U_W_LO_LIM = 0.8                    # Lower limit for filter to width control response
U_D_HI_LIM = 1.0                    # Upper limit for filter to height control response     
# ^ Max Scara allows is +/-1 for relative, unlimited in absolute
U_D_LO_LIM = -1.0                   # Lower limit for filter to height control response
SPEED      = float(750/600)         # speed in cm/s that the nozzle is moving
EXTRUSION_FACTOR_MAX = 2.0          # Maximum allowed Extrusion Factor
EXTRUSION_FACTOR_MIN = 0.5          # Minimum allowed Extrusion Factor
SPEED_FACTOR_MAX = 2.0              # Maximum allowed Speed Factor
SPEED_FACTOR_MIN = 0.5              # Minimum allowed Speed Factor
Z_BABY_MIN = -3.0                   # Minimum deviation allowed for baby stepping
Z_BABY_MAX = 3.0                    # Maximum deviation allowed for baby stepping
Z_BABY_STEP_MODE = 1                # 1 is relative baby steps, 0 is absolute baby steps 

TIME_STEP_THRESH = 3.0              # If greater than this time step between seeing valid frames, discount accumulated error
TIME_STEP_THRESH_DISCOUNT = 4.0     # Factor to discount by
WIDTH_CONTROL_QUEUE_L = 3           # Length of the averaging width queue before applying value
DEPTH_CONTROL_QUEUE_L = 5           # Length of the averaging depth queue before applying value
DEPTH_MEAN_DEV = 0.075              # maximum deviation from the mean permitted in queue
WIDTH_MEAN_DEV = 0.075              # maximum deviation from the mean permitted in queue
DEPTH_CONTROL_MIN_U = 0.1           # minimum of abs of input for the signal to be sent to printer (otherwise delay makes it not worth sending)

### EXPERIMENT PLOTTING VARIABLES
WIDTH_DEVIATION = 3                 # deviation in width from the target value when resetting
DEPTH_DEVIATION = 3                 # deviation in depth from the target value when resetting

### Below variables are only when referencing from video files
VIDEO_START_IDX = 0

class Feedback:

    def __init__(self, from_video, rec_video, video_name=None, rec_vid_name=None):
        self.acc_error_w = 0.0      # accumulated error in the width
        self.acc_error_d = 0.0      # accumulated error in the height
        self.correct_frames = 0     # num of frames that we analysed that were successful
        self.frames_checked = 0     # num of frames we tried to analyse regardless of result
        # self.im = image_module.ImageModule()
        self.kp_w = FEEDBACK_KP_W
        self.ki_w = FEEDBACK_KI_W
        self.kd_w = FEEDBACK_KD_W
        self.kp_d = FEEDBACK_KP_D
        self.ki_d = FEEDBACK_KI_D
        self.kd_d = FEEDBACK_KD_D
        self.t_layer_w = TARGET_LAYER_WIDTH
        self.t_layer_D = TARGET_LAYER_DEPTH

        self.width_hist = []                    # Tracks the history of the width updates
        self.depth_hist = []                    # Tracks the history of the depth updates
        self.width_dev_hist = []                # Tracks the history of the width error
        self.depth_dev_hist = []                # Tracks the historu of the depth error
        self.width_ref_line = []                # Holds reference line values for plotting
        self.depth_ref_line = []                # Holds reference line values for plotting
        self.depth_control_queue = []           # Queue for sending depth input to printer
        self.width_control_queue = []           # Queue for sending width input to printer
        self.depth_time_stamps = []             # Tracks timestamps for depth updates for plotting purposes
        self.width_time_stamps = []             # Tracks timestamps for width updates for plotting purposes 
        
        self.E_factor_hist = [100.0]            # Tracks the history of the E updates
        self.F_factor_hist = [100.0]            # Tracks the history of the F updates
        self.EF_ref_val = [100.0]               # Holds reference line values for plotting
        self.z_baby_hist = [0.0]                # Tracks the history of the z baby updates
        self.z_baby_ref_val = [0.0]             # Holds reference line values for plotting
        self.z_baby_time_stamps = [0.0]         # Tracks the times at which the z baby updates
        self.EF_factor_time_stamps = [0.0]      # Tracks the times at which the E/F factors update

        # Create and edit automatic plot data below
        plt.ion()
        self.depth_track_fig = plt.figure()
        self.depth_track_ax = self.depth_track_fig.add_subplot(111)
        self.depth_track_ax.set_title("Static Depth Measurements for {}mm Width Post Calibration".format(TARGET_LAYER_WIDTH))
        self.depth_track_ax.set_xlabel("Time (s)")
        self.depth_track_ax.set_ylabel("Error from Target Value (mm)")
        
        self.width_track_fig = plt.figure()
        self.width_track_ax = self.width_track_fig.add_subplot(111)
        self.width_track_ax.set_title("Static Width with Target={}mm Post Width Calibration".format(TARGET_LAYER_WIDTH))
        self.width_track_ax.set_xlabel("Time (s)")
        self.width_track_ax.set_ylabel("Error from Target Value (mm)")

        self.EF_track_fig = plt.figure()
        self.EF_track_ax = self.EF_track_fig.add_subplot(111)
        self.EF_track_ax.set_title("Width Control Feedback Inputs with Target={}mm".format(TARGET_LAYER_WIDTH))
        self.EF_track_ax.set_xlabel("Time (s)")
        self.EF_track_ax.set_ylabel("Mulitpliers (" + "%" + ")")
        self.EF_track_ax.set_ylim([EXTRUSION_FACTOR_MIN, EXTRUSION_FACTOR_MAX])

        self.z_baby_track_fig = plt.figure()
        self.z_baby_track_ax = self.z_baby_track_fig.add_subplot(111)
        self.z_baby_track_ax.set_title("Depth Control Feedback Inputs with Target={}mm".format(TARGET_LAYER_WIDTH))
        self.z_baby_track_ax.set_xlabel("Time (s)")
        self.z_baby_track_ax.set_ylabel("Z Baby Step Value (mm)")
        self.z_baby_track_ax.set_ylim([Z_BABY_MIN, Z_BABY_MAX])

        self.speed_factor = 1.0         # Speed multiplier recorded in SCARA
        self.extrusion_factor = 1.0     # Extrusion multiplier recorded in SCARA
        self.z_baby = 0.0               # Z baby steps off the starting homed value

        self.from_video = from_video
        self.im = ImageModule()

        if from_video:
            if video_name is None:
                raise RuntimeError("`video_name` cannot be None when drawing from video")
            self.__init_video__(video_name)
        else:
            self.cam = IntelD405()
            self.printer = DWA('http://Scara')

        if rec_video:
            self.init_video_record(output_vid_name=REC_VIDEO_NAME)

        self.start_time = time.time()       # Make it at end of init so time reflects start of feedback time
        self.ignore_time = 0.0

    def __init_video__(self, video_name):
        """
        Initializes the video for reading from later instead of real time
        """

        color_video_name = video_name + "_color.avi"
        depth_video_name = video_name + "_depth.avi"
        depth_data_file_name = video_name + "_depth.pkl"
        color_cap = cv2.VideoCapture(color_video_name)
        depth_cap = cv2.VideoCapture(depth_video_name)
        with open(depth_data_file_name, "rb") as fp:   # Unpickling
            self.depth_data = pickle.load(fp)

        # Collect RGB Frame information in advance to process simultaneously
        self.color_frames = []
        counter = 0
        while color_cap.isOpened():
            ret, color_frame = color_cap.read()
            if ret == False:
                raise RuntimeError("Can't read frames for some reason. Exiting")
            self.color_frames.append(color_frame)

            if counter == len(self.depth_data)-1:
                break

            counter = counter + 1
        color_cap.release()

        # Collect Colorized Depth Frame information in advance to process simultaneously
        self.colorized_depth_frames = []
        counter = 0
        while depth_cap.isOpened():
            ret, depth_frame = depth_cap.read()
            if ret == False:
                raise RuntimeError("Can't read frames for some reason. Exiting")
            self.colorized_depth_frames.append(depth_frame)

            if counter == len(self.depth_data)-1:
                break

            counter = counter + 1
        depth_cap.release()

        self.video_idx = VIDEO_START_IDX

        return
    
    def init_video_record(self, output_vid_name):
        """
        Initializes the recording process for a storing a video of the results of the experiment
        """

        self.color_rec_vid_file_name = output_vid_name + "_color.avi"
        self.depth_rec_vid_file_name = output_vid_name + "_depth.avi"
        self.depth_data_file_name = output_vid_name + "_depth.pkl" 

        fps = 30
        im_size = (480, 640)

        self.color_rec_video = cv2.VideoWriter(self.color_rec_vid_file_name,cv2.VideoWriter_fourcc('M','J','P','G'), fps, im_size, True)
        self.depth_rec_video = cv2.VideoWriter(self.depth_rec_vid_file_name,cv2.VideoWriter_fourcc('M','J','P','G'), fps, im_size, True)
        self.depth_rec_raw_data = []

    def update_rec_video(self, colorized_depth_frame, color_frame, raw_depth):
        """
        Add frames to the recorded video
        """
        if RECORD_VIDEO:
            self.depth_rec_video.write(colorized_depth_frame)
            self.color_rec_video.write(color_frame)
            self.depth_rec_raw_data.append(raw_depth)

            return True
        return False
    
    def save_rec_video(self):
        """
        Saves the output of the recorded video as a color video, a
        colorized depth video, and a pickled version of the raw depth data
        """
        self.color_rec_video.release()
        self.depth_rec_video.release()

        with open(self.depth_data_file_name, "wb") as depth_file:   #Pickling
            pickle.dump(self.depth_rec_raw_data, depth_file)

    def teardown(self):
        """
        Call at the end of the feedback loop. Makes sure everything concludes correctly
        """

        print("----------- END OF FEEDBACK -----------")
        print("Feedback concluded after {} seconds of operation".format(np.round(self.timecheck())))
        print("Successful Frames Ratio = {}".format(self.get_frame_analysis_success_ratio()))

        # To have lines go to the end of the plot
        self.E_factor_hist.append(np.round(self.extrusion_factor * 100.0, 3))
        self.F_factor_hist.append(np.round(self.speed_factor * 100.0, 3))
        self.EF_ref_val.append(100)
        self.EF_factor_time_stamps.append(time.time() - self.start_time - self.ignore_time)

        self.z_baby_hist.append(np.round(self.z_baby, 3))
        self.z_baby_ref_val.append(0.0)
        self.z_baby_time_stamps.append(time.time() - self.start_time - self.ignore_time)

        if len(self.width_hist) > 0:
            print("Saving Width Plot")
            self.save_rt_plots(width=True)
        if len(self.depth_hist) > 0:
            print("Saving Depth Plot")
            self.save_rt_plots(depth=True)
        if len(self.E_factor_hist) > 2:
            print("Saving EF Plot")
            self.save_rt_plots(EF=True)
        if len(self.z_baby_hist) > 2:
            print("Saving ZB Plot")
            self.save_rt_plots(z_baby=True)

        print("Saving data")
        ddata = np.vstack((self.depth_hist, self.depth_dev_hist, self.depth_time_stamps, self.depth_ref_line))
        with open("depth_data_{}.pkl".format(int(time.time())), "wb") as depth_file:   #Pickling
            pickle.dump(ddata, depth_file)
        wdata = np.vstack((self.width_hist, self.width_dev_hist, self.width_time_stamps, self.width_ref_line))
        with open("width_data_{}.pkl".format(int(time.time())), "wb") as width_file:   #Pickling
            pickle.dump(wdata, width_file)
        EFdata = np.vstack((self.E_factor_hist, self.F_factor_hist, self.EF_factor_time_stamps, self.EF_ref_val))
        with open("EF_data_{}.pkl".format(int(time.time())), "wb") as EF_file:   #Pickling
            pickle.dump(EFdata, EF_file)
        ZBdata = np.vstack((self.z_baby_hist, self.z_baby_time_stamps, self.z_baby_ref_val))
        with open("ZB_data_{}.pkl".format(int(time.time())), "wb") as ZB_file:   #Pickling
            pickle.dump(ZBdata, ZB_file)

        if RECORD_VIDEO:
            print("Saving Recorded Video. This may take a moment.")
            self.save_rec_video()

        print("---------- TEARDOWN COMPLETE ----------")

    
    def update_rt_plots(self, depth=False, width=False, EF=False, z_baby=False):
        """
        Update the real-time plots to reflect most recent data
        Returns True if a value was updated. Returns false if not
        """
        updated = False
        if len(self.depth_hist) > 0 and depth:
            self.depth_track_ax.plot(self.depth_time_stamps, self.depth_ref_line, color="orange", linestyle="dashed")
            self.depth_track_ax.plot(self.depth_time_stamps, self.depth_dev_hist, "b-")
            self.depth_track_fig.canvas.draw()
            plt.pause(0.1)
            updated = True
        if len(self.width_hist) > 0 and width:
            self.width_track_ax.plot(self.width_time_stamps, self.width_ref_line, color="orange", linestyle="dashed")
            self.width_track_ax.plot(self.width_time_stamps, self.width_dev_hist, "r-")
            self.width_track_fig.canvas.draw()
            plt.pause(0.1)
            updated = True
        if len(self.E_factor_hist) > 0 and EF:
            self.EF_track_ax.plot(self.EF_factor_time_stamps, self.EF_ref_val, color="orange", linestyle="dashed")
            self.EF_track_ax.plot(self.EF_factor_time_stamps, self.E_factor_hist, color="red")
            self.EF_track_ax.plot(self.EF_factor_time_stamps, self.F_factor_hist, color="blue")
            self.EF_track_fig.canvas.draw()
            plt.pause(0.1)
            updated = True
        if len(self.z_baby_hist) > 0 and z_baby:
            self.z_baby_track_ax.plot(self.z_baby_time_stamps, self.z_baby_ref_val, color="orange", linestyle="dashed")
            self.z_baby_track_ax.plot(self.z_baby_time_stamps, self.z_baby_hist, color="green")
            self.z_baby_track_fig.canvas.draw()
            plt.pause(0.1)
            updated = True

        return updated
    
    def save_rt_plots(self, depth=False, width=False, EF=False, z_baby=False):
        """
        Saves the real time plots to file in order to reset code and run new experiment
        """
        if depth:
            if len(self.depth_hist) == 0:
                raise RuntimeError("No depth history so no depth plot to save")
            
            self.depth_track_fig.savefig("depth_deviation_{}.png".format(int(time.time())), format='png', dpi=300)

        if width:
            if len(self.width_hist) == 0:
                raise RuntimeError("No width history so no width plot to save")
            
            self.width_track_fig.savefig("width_deviation_{}.png".format(int(time.time())), format='png', dpi=300)

        if EF:
            if len(self.E_factor_hist) == 0:
                raise RuntimeError("No E/F history so no EF plot to save")
            
            self.EF_track_fig.savefig("EF_updates_{}.png".format(int(time.time())), format='png', dpi=300)

        if z_baby:
            if len(self.z_baby_hist) == 0:
                raise RuntimeError("No width history so no width plot to save")
            
            self.z_baby_track_fig.savefig("z_baby_updates_{}.png".format(int(time.time())), format='png', dpi=300)

    
    def get_video_frames(self):
        """
        If drawing from recorded video, it supplies a single timestep of frames from video
        """
        raw_depth_frame = self.depth_data[self.video_idx].__deepcopy__(self.depth_data[self.video_idx])
        color_frame = self.color_frames[self.video_idx].__deepcopy__(self.color_frames[self.video_idx])
        colorized_depth_frame = self.colorized_depth_frames[self.video_idx].__deepcopy__(self.colorized_depth_frames[self.video_idx])
        self.video_idx += 1
        return raw_depth_frame, color_frame, colorized_depth_frame

    def timecheck(self) -> float:
        """
        Returns the time in seconds since the start time
        """
        return time.time() - self.start_time
    

    def adjust_camera(self, relative_offset):        
        """
        baby-steps the camera in order to have it more closely align to the desired path
        """
        gcode_line = "M290 C{} R1".format(relative_offset)
        if self.from_video == False:
            self.printer.send_code(gcode_line)
        return
    
    def adjust_x(self, relative_offset):
        """
        baby-steps the x-axis in order to adjust the camera perspective
        """
        gcode_line = "M290 X{} R1".format(relative_offset)
        if self.from_video == False:
            self.printer.send_code(gcode_line)
        return

    def adjust_y(self, relative_offset):
        gcode_line = "M290 Y{} R1".format(relative_offset)
        if self.from_video == False:
            self.printer.send_code(gcode_line)
        return

    def get_frame_analysis_success_ratio(self):
        """
        Returns the ratio of frames checked frames 
        """
        print("Frames Checked = {}; Correct Frames = {}".format(self.frames_checked, self.correct_frames))
        return np.round(self.correct_frames / self.frames_checked, 5)
    
    def update_frame_analysis(self, success):
        """
        Updates the count for the number of frames that have been checked
        and the number of successful frames checked. 
        """
        self.frames_checked = self.frames_checked + 1
        if success:
            self.correct_frames = self.correct_frames + 1

    def send_gcode_update(self, u_w, u_d):
        """
        Takes the calculated control inputs from the PID loop and sends
        appropriate update to the Duet board to actuate the change. 
        """

        print("u_w = {};   u_d = {}".format(np.round(u_w, 3), np.round(u_d, 3)))
        
        # Speed and Extrusion factor should always be linked for this program's purposes
        # Only do checks on the one and the other will follow
        if self.extrusion_factor * u_w >= SPEED_FACTOR_MAX:
            u_w = np.round(SPEED_FACTOR_MAX/self.extrusion_factor, 4)
        elif self.extrusion_factor * u_w <= SPEED_FACTOR_MIN:
            u_w = np.round(SPEED_FACTOR_MIN/self.extrusion_factor, 4)
            
        self.width_control_queue.append(np.round(u_w, 4))
        self.depth_control_queue.append(np.round(u_d, 4))
        
        # Remove any that deviate largely from the average values stored in the queue
        if len(self.width_control_queue) >= WIDTH_CONTROL_QUEUE_L:
            mean = np.mean(self.width_control_queue)
            self.width_control_queue = [k for k in self.width_control_queue if k < mean + WIDTH_MEAN_DEV and k > mean - WIDTH_MEAN_DEV]
        
        # When the queue is the proper size all within a stddev, then apply value
        if len(self.width_control_queue) >= WIDTH_CONTROL_QUEUE_L:
            self.extrusion_factor = np.round(self.extrusion_factor * np.mean(self.width_control_queue), 4)
            self.speed_factor = np.round(1/self.extrusion_factor, 4)
            self.width_control_queue = []
            # To make sections representative add 2 values, one at previous value and one at next value
            # Previous value:
            self.E_factor_hist.append(self.E_factor_hist[-1])
            self.F_factor_hist.append(self.F_factor_hist[-1])
            self.EF_ref_val.append(100)
            self.EF_factor_time_stamps.append(time.time() - self.start_time - self.ignore_time)
            # New value
            self.E_factor_hist.append(np.round(self.extrusion_factor * 100.0, 3))
            self.F_factor_hist.append(np.round(self.speed_factor * 100.0, 3))
            self.EF_ref_val.append(100)
            self.EF_factor_time_stamps.append(time.time() + 0.001 - self.start_time - self.ignore_time)
            if not self.from_video: 
                print("Extrusion Factor changed to = {}".format(self.extrusion_factor))
                print("Speed Factor changed to = {}".format(self.speed_factor))
                # Send G-code
                self.printer.send_code("M220 S{}".format(self.speed_factor*100.0))   # Overwrite speed factor
                # # S is the new speed factor
                self.printer.send_code("M221 S{}".format(self.extrusion_factor*100.0))   # Overwrite extrusion factor percentage
                # S is the new extrusion factor

        # Remove any that deviate largely from the average values stored in the queue
        if len(self.depth_control_queue) >= DEPTH_CONTROL_QUEUE_L:
            mean = np.mean(self.depth_control_queue)
            self.depth_control_queue = [k for k in self.depth_control_queue if k < mean + DEPTH_MEAN_DEV and k > mean - DEPTH_MEAN_DEV]

        # After reaching min requirement for an appropriate average 
        if len(self.depth_control_queue) >= DEPTH_CONTROL_QUEUE_L:
            if np.abs(np.mean(self.depth_control_queue)) > DEPTH_CONTROL_MIN_U: 
                self.z_baby = self.z_baby + np.mean(self.depth_control_queue)
                self.depth_control_queue = []
                # To make sections representative add 2 values, one at previous value and one at next value
                # Previous value:
                self.z_baby_hist.append(self.z_baby_hist[-1])
                self.z_baby_ref_val.append(0.0)
                self.z_baby_time_stamps.append(time.time() + 0.001 - self.start_time - self.ignore_time)
                # New value
                self.z_baby_hist.append(np.round(self.z_baby, 3))
                self.z_baby_ref_val.append(0.0)
                self.z_baby_time_stamps.append(time.time() - self.start_time - self.ignore_time)
                if not self.from_video:
                    # Send G-code
                    self.printer.send_code("M290 S{} R{}".format(np.round(u_d, 4), Z_BABY_STEP_MODE))
                    print("Z baby step changed to = {}".format(self.z_baby))
            else:
                self.depth_control_queue = []

        return True

    def pid_feedback_loop(self, last_error_w, last_error_d, time_last_data):
        this_loop_ignore_time = 0.0
        it_time = time.time()

        kp_w = FEEDBACK_KP_W
        ki_w = FEEDBACK_KI_W
        kd_w = FEEDBACK_KD_W

        kp_d = FEEDBACK_KP_D
        ki_d = FEEDBACK_KI_D
        kd_d = FEEDBACK_KD_D

        t_layer_w = TARGET_LAYER_WIDTH
        t_layer_d = TARGET_HEIGHT_ABOVE_FIL

       # While there is not a successful frame to process one, try to collect one
        successful_frame = False
        while successful_frame == False:
            time_start_frame_search = time.time()
            if self.from_video:
                if self.video_idx == len(self.depth_data)-1:
                    return None, None, None, None, None
                raw_depth_frame, color_frame, colorized_depth_frame = self.get_video_frames()
                if RECORD_VIDEO:
                    self.update_rec_video(colorized_depth_frame=colorized_depth_frame, 
                                          color_frame=color_frame, 
                                          raw_depth=raw_depth_frame)
            else:
                suc, raw_depth_frame, color_frame = self.cam.get_frames(as_array=True, keep=RECORD_VIDEO)
                colorized_depth_frame = color_frame 
                if RECORD_VIDEO:
                    self.update_rec_video(colorized_depth_frame=colorized_depth_frame, 
                                          color_frame=color_frame, 
                                          raw_depth=raw_depth_frame)

            # process the frame
            proc_color, proc_depth, proc_raw_depth = self.im.process_frame(color_frame=color_frame,
                                                                      colorized_depth_frame=colorized_depth_frame,
                                                                      raw_depth_frame=raw_depth_frame)
            outlined_img = self.im.outline_filament2(proc_color)
            ret, width, depth, width_im, width_slope = self.im.calc_filament_props(outlined_frame=outlined_img, color_frame=proc_color, 
                                                                              raw_filtered_depth=proc_raw_depth, speed=SPEED)

            if ret:
                a_layer_w = width       # actual measured layer width    (measured value)
                a_layer_d = depth       # actual measured layer height   (measured value)
                successful_frame = True
                self.update_frame_analysis(success=True)
                cv2.imshow("Frame 1", color_frame)
                cv2.imshow("Frame 2", colorized_depth_frame)
            else:
                # cut this search time from plots and accumulated error calcs
                self.ignore_time += time.time() - time_start_frame_search
                this_loop_ignore_time += time.time() - time_start_frame_search
                print("Failing to find appropriate frame")
                self.update_frame_analysis(success=False)
                cv2.imshow("Frame 1", color_frame)
                cv2.imshow("Frame 2", outlined_img)
        
        time_received = time.time()      # Time the last data point was received in seconds
        TIME_STEP = time_received - time_last_data - this_loop_ignore_time

        error_w = a_layer_w - t_layer_w
        error_d = a_layer_d - t_layer_d

        self.width_hist.append(a_layer_w)
        self.width_dev_hist.append(error_w)
        self.width_time_stamps.append(time_received - self.start_time - self.ignore_time)
        self.width_ref_line.append(0.0)
        self.depth_hist.append(a_layer_d)
        self.depth_dev_hist.append(error_d)
        self.depth_time_stamps.append(time_received - self.start_time - self.ignore_time)
        self.depth_ref_line.append(0.0)

        error_w_slope = (error_w - last_error_w)/TIME_STEP
        error_d_slope = (error_d - last_error_d)/TIME_STEP

        self.acc_error_w += error_w * TIME_STEP
        self.acc_error_d += error_d * TIME_STEP

        error_w_integral = self.acc_error_w 
        error_d_integral = self.acc_error_d

        # Control System Response (width management)
        u_w = (kp_w*error_w) + (ki_w*error_w_integral) + (kd_w*error_w_slope)

        # Makes sure that negative errors result in fractions < 1 and
        # positive errors result in fractions > 1 
        u_w = 1 + u_w
        
        # Control System Response (height management)
        u_d = (kp_d*error_d) + (ki_d*error_d_integral) + (kd_d*error_d_slope)

        # Apply filters to the response:

        # Limit filters -- These are to prevent too strong responses which will be harder to recover from
        u_w = min(u_w, U_W_HI_LIM)
        u_w = max(u_w, U_W_LO_LIM)

        u_d = min(u_d, U_D_HI_LIM)
        u_d = max(u_d, U_D_LO_LIM)

        # Translate controls to appropriate G-code and send to printer

        self.send_gcode_update(u_w, u_d)

        blank_img = blank_img = np.zeros((color_frame.shape[0], color_frame.shape[1]))
        on_black, on_color = self.im.collate(width_im, blank_img, color_frame=color_frame)

        frame_show1 = on_color
        frame_show2 = outlined_img

        self.update_rt_plots(depth=True)
        self.update_rt_plots(width=True)
        self.update_rt_plots(z_baby=True)
        self.update_rt_plots(EF=True)

        return error_w, error_d, time_received, frame_show1, frame_show2

    
def main():

    fdbk = Feedback(from_video=FROM_VIDEO, video_name=READ_VIDEO_NAME, 
                    rec_video=RECORD_VIDEO, rec_vid_name=READ_VIDEO_NAME)
    
    last_error_w = 0.0; last_error_d = 0.0; time_last_data = time.time()

    while True:
        
        last_error_w, last_error_d, time_last_data, frame1, frame2 = fdbk.pid_feedback_loop(last_error_w, last_error_d, time_last_data)
        if last_error_w is None:
            break
        if time.time() - fdbk.start_time - fdbk.ignore_time > 300:
            break

        key_press = cv2.waitKey(1)
        if key_press == ord('x'):
            break
        if key_press == ord('1'):
            fdbk.adjust_camera(1)       # Moves 1 degree CW  (1 C)
        if key_press == ord('2'):       
            fdbk.adjust_camera(-1)      # Moves 1 degree CCW (2 C's)
        if key_press == ord('w'):
            fdbk.adjust_y(-0.25)        # Move Scara forward
        if key_press == ord('s'):
            fdbk.adjust_y(0.25)         # Move Scara Backward
        if key_press == ord('a'):
            fdbk.adjust_x(-0.25)        # Move Scara left
        if key_press == ord('d'):
            fdbk.adjust_x(0.25)         # Move Scara Right

    fdbk.teardown()
    return

# Runs the main method upon execution of the file
if __name__ == '__main__': 
    main()