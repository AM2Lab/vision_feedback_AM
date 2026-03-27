##############################################################
#
#  Performs real time feedback control 
#  calls: IntelD405() from camera.py
#  Code Created by Benjamin Gorse for Princeton 2024 Sr Thesis
#
##############################################################

import cv2
import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
from camera import IntelD405
import os
import time
import queue
import pickle
import scipy
import time

PATH_HERE = os.path.dirname(os.path.abspath(__file__))
PATH_EXPORT = os.path.join(PATH_HERE, 'video_outputs\\test_timed_video.avi')

# initialize an arbitrary value
mouse_point = (100,200)

def get_mouse_info(event, x, y, args, params):
    global mouse_point
    mouse_point = (x, y)

class ImageModule():

    def __init__(self):
        self.FOV_H_DEG = 87
        self.FOV_V_DEG = 58 
        self.FOV_H_RAD = self.FOV_H_DEG * np.pi / 180.0
        self.FOV_V_RAD = self.FOV_V_DEG * np.pi / 180.0
        self.time_zero = time.time()

        # Graph recording values:
        # Depth graph values
        self.depth_graph_data = []
        self.depth_independent_data = []
        self.depth_graph_err_data = []
        self.depth_graph_max_data = []
        self.depth_graph_min_data = []

        # Monitoring the depth control feedback
        self.depth_control_graph_data = []
        self.depth_control_independent_data = []
        self.depth_control_graph_err_data = []

        # Depth of filament values
        self.fil_depth_graph_data = []
        self.fil_width_graph_data = []
        self.fil_depth_independent_data = []
        self.fil_width_compare_data = []            # This is to compare width from constant depth calc to camera measured depth calc

        return

    def show_blurs(self, cam, d, sigma_Color, sigma_Space):
        """
        Shows a blurred version of the image in comparison with the unmodified image. 
        `d` sets the height/width of the patch that is blurred
        `sigma_Color` is the std dev between the colors (how diff are colors)
        `sigma_Space` changes the weights depending on the distance of the pixel from the target pixel
        large sigmas (>100) will have a stronger filter effect
        use `d`=5 for good real time speed performance
        """
        while True:
            success, depth, color = cam.get_frames(as_array=True)
            if success:
                new_image = cv2.bilateralFilter(color, d, sigmaColor=sigma_Color, sigmaSpace=sigma_Space)
                cv2.imshow("blurred image", new_image)
                cv2.imshow("original image", color)
            else:
                print("FAILED FRAME")

            # Hit q to quit.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def bilateral_blur_frame(self, frame, d, sigma_color, sigma_space):
        """
        Blurs a single frame using a bilateral filter and returns it

        Parameters:
            frame : image frame
                Image to be blurred
            d : 
                something
            sigma_color : int

            sigma_space : int
        """
        return cv2.bilateralFilter(frame, d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    
    def __update_all__(self, img, px_list, value):
        """
        Updates all pixels provided at the coords in `px_list` for the image `img` to be `value`
        """
        for px in px_list:
            img[px[0], px[1]] = value
    
    def __sort_by_x__(self, px):
        return px[1]
    
    def __sort_by_y__(self, px):
        return px[0]

    def outline_filament(self, proc_color_frame):
        """
        Takes in frames from self.process_frames()

        Returns frames with outlined edge of the filament 
        """
        BLACK = 0
        WHITE = 255
        SLOPE_THRESH = 0.25

        proc_color_frame = proc_color_frame.__deepcopy__(proc_color_frame)
        y_size = proc_color_frame.shape[0]
        center_y = int(240/y_size)
        x_size = proc_color_frame.shape[1]
        center_x = int(240/y_size)
        section_w = 5

        im_shape = proc_color_frame.shape

        # get indices of white mask:
        white_mask = np.argwhere(proc_color_frame == WHITE)     

        # must be odd
        check_width = 5
        check_height = 5
        check_px = check_height*check_width
        # extend beyond center
        w_ext = int((check_width - 1)/2)
        h_ext = int((check_height - 1)/2)

        # wider 2nd check
        check2_width = 11
        check2_height = 9
        check2_px = check2_height*check2_width

        # Determines if there are a minimum of pixels close
        filtered_wm = [k for k in white_mask if (k[0] >= h_ext) and (k[1] >= w_ext) and (k[0] < y_size-h_ext) and (k[1] < x_size-w_ext) and
                       np.count_nonzero(proc_color_frame[(k[0]-h_ext):(k[0]+h_ext), (k[1]-w_ext):(k[1]+w_ext)]) >= check_width]
        
        new_im = np.zeros(im_shape)
        for px in filtered_wm:
            new_im[px[0], px[1]] = 255

        # Assigns unique id's to each separate foreground (white) element
        success, markers = cv2.connectedComponents(new_im.astype(np.uint8))

        num_markers = len(np.unique(markers))

        # Get the pixels at the end of each region of interest (specifically x-extremes)
        global extreme_px_coords     
        extreme_px_coords = []       # Will store coords of the nearest points
        approx_slopes = []           # Will store the approximate slopes of the roi's
        for mrk in range(num_markers):
            if mrk == 0:
                continue # don't count the background
            roi_np = np.argwhere(markers == mrk)   # region of interest
            roi = [k for k in roi_np]
            roi.sort(key=self.__sort_by_x__)
            extreme_px_coords.append(roi[0])        
            extreme_px_coords.append(roi[-1])        
            y1,x1 = roi[0]                  # px of smallest x coordinate in ROI
            y2,x2 = roi[-1]                 # px of largest x coordinate in ROI
            # determine approximate slope of the MARKED ROI
            if len(roi) <= 1:               # SINGLE POINT in ROI
                approx_slopes.append(int(1))  
                approx_slopes.append(int(1))
                continue
            if x1 == x2:                    # VERTICAL LINE 
                approx_slopes.append(int(2))  
                approx_slopes.append(int(2))
                continue
            roi_x = roi_np[:,1]
            roi_y = roi_np[:,0]
            slope, offset = np.polyfit(roi_x, roi_y, deg=1)
            approx_slopes.append(float(slope))  # Should have inverse of slope because goes other direction to left
            approx_slopes.append(float(slope))
            

        # Use K-d tree to find nearest neighbors
        if len(extreme_px_coords) >= 2:          # Requires k-d tree to have at least 2 dimensions
            tree = scipy.spatial.KDTree(data=extreme_px_coords)
            radius = 50
            num_to_search_to = 10
            dists, near_pxs = tree.query(x=extreme_px_coords, k=num_to_search_to, distance_upper_bound=radius) 
            # near_pxs is sorted in by proximity in K-d tree

            for ex_idx, near_array in enumerate(near_pxs):
                mrk_idx = np.floor(ex_idx/2) + 1
                LEFT = 0
                RIGHT = 1

                ### TEST CODE START
                newer_img = np.zeros(im_shape)
                newer_img[markers!=0] = 50
                ### TEST CODE END

                y1,x1 = extreme_px_coords[ex_idx]

                counter = 0
                closest_idx = -1
                found_px = False
                # Get the closest valid pixel
                while counter < len(near_array):
                    # print("start loop with: mrk_idx={}, ex_idx={}, counter={}, near_array[counter] = {}, len(near_array)={}".format(mrk_idx, ex_idx, counter, near_array[counter], len(near_array)))
                    if mrk_idx == np.floor(near_array[counter]/2) +1:
                        counter = counter + 1
                        continue
                    if near_array[counter] == len(extreme_px_coords):
                        # This is condition where there is no closest pixel within range 
                        break
                    # next verify that the line is leaving the extreme in the correct direction
                    # and verify that the line is connecting on the correct extreme of other ROI
                    y2,x2 = extreme_px_coords[near_array[counter]]
                    if ex_idx % 2 == LEFT:                
                        if x2 > x1:                 # line must be left of leftmost x of ROI
                            counter = counter + 1
                            continue
                        if near_array[counter] % 2 == LEFT:
                            counter = counter + 1
                            continue
                    else:                       
                        if x2 < x1:                 # line must be right of rightmost x of ROI
                            counter = counter + 1
                            continue
                        if near_array[counter] % 2 == RIGHT:
                            counter = counter + 1
                            continue
                    
                    
                    closest_idx = near_array[counter]
                    found_px = True
                    break
                # Skip iteration if there is no target point to draw a line to
                if found_px == False:
                    continue

                
                
                line_length = np.abs(y2-y1)

                # Check if it is valid to draw the line for the pixel:
                if type(approx_slopes[ex_idx]) == int:      # aka precalc-ed slope is a vertical line/single point by above convention
                    if approx_slopes[ex_idx] == 1:          # Single point in ROI 
                        if pow(y2-y1,2)-0.5*pow(x2-x1,2) < 0:
                            cv2.line(img=new_im, pt1=(extreme_px_coords[ex_idx][1],extreme_px_coords[ex_idx][0]) , 
                                    pt2=(extreme_px_coords[closest_idx][1], extreme_px_coords[closest_idx][0]), 
                                    color=(200,200,200), thickness=1)
                        else:
                            pass
                    elif approx_slopes[ex_idx] == 2:        # ROI is a vertical line
                        if x2-x1 != 0 and pow(y2-y1,2)-0.5*pow(x2-x1,2) < 0:
                            cv2.line(img=new_im, pt1=(extreme_px_coords[ex_idx][1],extreme_px_coords[ex_idx][0]) , 
                                    pt2=(extreme_px_coords[closest_idx][1], extreme_px_coords[closest_idx][0]), 
                                    color=(150,150,150), thickness=1)

                else:
                    if x2-x1 == 0:                          # aka line between endpoints would be vertical 
                        if line_length <= 3:
                            cv2.line(img=new_im, pt1=(extreme_px_coords[ex_idx][1],extreme_px_coords[ex_idx][0]) , 
                                    pt2=(extreme_px_coords[closest_idx][1], extreme_px_coords[closest_idx][0]), 
                                    color=(100,100,100), thickness=1)
                            
                    else:
                        line_slope = (y2-y1)/(x2-x1)
                        if pow(y2-y1,2)-0.5*pow(x2-x1,2) <= 0 and np.abs(approx_slopes[ex_idx]-line_slope) <= SLOPE_THRESH:      # Slope is below margin AND close to other slope
                            cv2.line(img=new_im, pt1=(extreme_px_coords[ex_idx][1],extreme_px_coords[ex_idx][0]) , 
                                    pt2=(extreme_px_coords[closest_idx][1], extreme_px_coords[closest_idx][0]), 
                                    color=(50,50,50), thickness=1)

        # After filling in lines, remove elements of low aspect ratio AND elements of <1% of foreground pixels
        # Assigns unique id's to each separate foreground (white) element
        MIN_PX_RATIO_THRESH = 0.05
        MAX_AR_RATIO_THRESH = 0.8
        success, markers = cv2.connectedComponents(new_im.astype(np.uint8))

        num_markers = len(np.unique(markers))

        rois = []
        roi_lens = []
        roi_ARs = []
        max_AR = 1
        for mrk in range(num_markers):
            if mrk == 0:
                continue # don't count the background
            roi_np = np.argwhere(markers == mrk)   # region of interest
            rois.append(roi_np)
            roi_lens.append(len(roi_np))

            roi = [k for k in roi_np]
            roi.sort(key=self.__sort_by_x__)
            width = np.abs(roi[-1][1] - roi[0][1])
            roi.sort(key=self.__sort_by_y__)
            height = np.abs(roi[-1][0] - roi[0][0])
            if height == 0:
                roi_ARs.append(-1)
                continue
            AR = width/height
            roi_ARs.append(AR)
            max_AR = max(max_AR, AR)

        # Replace -1 flags with max_AR value so horizontal lines remain
        roi_ARs = [k if k!=-1 else max_AR for k in roi_ARs]

        tot_fg_pxs = np.sum(roi_lens)

        for idx, roi_np in enumerate(rois):
            # Removes foreground patches below the number of pixels threshold
            if roi_lens[idx] / tot_fg_pxs <= MIN_PX_RATIO_THRESH:
                new_im[markers == idx + 1] = 0
        
        return new_im

    def calc_filament_props(self, outlined_frame, color_frame, raw_filtered_depth, speed=None):
        """
        `speed` parameter is in mm/s

        Takes in a frame with only the outline of the filament centered around the y midline
        focusing on the right side of the frame (as that is the most recently printed material)

        Returns an estimate for the depth and the width 
        If the input frame is bad, returns None for the width. Otherwise, returns an estimate for the width
        """
        
        bad_counter = 0

        WHITE = 255
        get_y = 0
        get_x = 1

        # Start to the far right of the frame
        # check width range of 5 pixels at a time. Take the average value of the width for those values. 
        # If there is an issue, 
        im_shape = outlined_frame.shape
        y_size = im_shape[0]
        x_size = im_shape[1]
        center_y = int(y_size/2)
        center_x = int(x_size/2)
        section_w = 5
        sections_check = 8              # number of sections to check
        req_good_sections = 4           # number of sections that must be valid to be considered successful
        border_threshold = 10           # distance in px between RH edge of frame and location of start of analysis
        min_check_start = int(8*im_shape[1]/10)      # minimum distance (in px) can explore to the left before giving up
        max_height_sep = 20             # max distance in px that the average of a range can be separated from extreme px
        width_px_thresh = 200           # max width that the filament can be (in px) before unreasonable to be filament width
        sect_deviation_thresh = 20      # maximum that an individual section can deviate from the mean of the sections


        range_end = x_size - 1 - border_threshold                       # Right most bound to section for analysis
        range_start = range_end - section_w + 1 - border_threshold      # Left most bound to section for analysis
        curr_x_idx = range_end                                          # Current position within section analysis
        sections_found = []                                             # Each valid recorded section (inclusive range)
        hi_section_candidates = []                                      # Tuple for start range idx, end range idx, ...
        lo_section_candidates = []                                      # Tuple for start range idx, end range idx, ...
        white_mask = np.argwhere(outlined_frame == WHITE)               # Gets the pixel coordinates for all white pixels 
        width_calc_overlay_im = np.zeros(im_shape)
        while range_start >= min_check_start and len(sections_found) < req_good_sections:

            # Get vertical range to check for valid pixels
            hi_candidate_px = [k for k in white_mask if k[get_x]==curr_x_idx and k[get_y] <  center_y]      # High in frame
            lo_candidate_px = [k for k in white_mask if k[get_x]==curr_x_idx and k[get_y] >= center_y]      # Low in frame

            # If lengths are zero, that means no pixels were found in valid range
            if len(hi_candidate_px) == 0 or len(lo_candidate_px) == 0:
                range_end = curr_x_idx - 1
                range_start = range_end - section_w + 1
                curr_x_idx = range_end
                continue

            # Gets single-pixel best candidate to be on the line
            hi_candidate_px = np.array([np.max(hi_candidate_px, axis=0)[get_y], curr_x_idx])
            lo_candidate_px = np.array([np.min(lo_candidate_px, axis=0)[get_y], curr_x_idx])

            if len(hi_section_candidates) == 0 or len(lo_section_candidates) == 0:                     # Should always be the same for both
                # If the list of candidates is empty, then nothing to compare to. Add it
                hi_section_candidates.append(hi_candidate_px)
                lo_section_candidates.append(lo_candidate_px)
            else:
                px_width = lo_candidate_px[get_y] - hi_candidate_px[get_y] 
                if px_width > width_px_thresh:
                    # If the width is unreasonably wide, it is probably a bad measurement
                    range_end = curr_x_idx - 1
                    range_start = range_end - section_w + 1
                    curr_x_idx = range_end
                    continue

                hi_ave = np.mean(hi_section_candidates, axis=0)[get_y]
                lo_ave = np.mean(lo_section_candidates, axis=0)[get_y]
                if hi_ave + max_height_sep < hi_candidate_px[get_y] or hi_ave - max_height_sep > hi_candidate_px[get_y] or \
                   lo_ave + max_height_sep < lo_candidate_px[get_y] or lo_ave - max_height_sep > lo_candidate_px[get_y]:
                    # If any of the candidates are really far away from the existing line, it is probably noise
                    range_end = curr_x_idx - 1
                    range_start = range_end - section_w + 1
                    curr_x_idx = range_end
                    continue

                if len(sections_found) != 0:
                    sect_width_mean = 0
                    for q in sections_found:
                        sect_width_mean = sect_width_mean + q[4]
                    sect_width_mean = sect_width_mean / len(sections_found)
                    this_sect_width = (lo_ave_y - hi_ave_y)
                    if np.abs(sect_width_mean - this_sect_width) > sect_deviation_thresh:
                        # if any candidate sections are really far from other sections: something is wrong
                        range_end = curr_x_idx - 1
                        range_start = range_end - section_w + 1
                        curr_x_idx = range_end
                        continue

                hi_section_candidates.append(hi_candidate_px)
                lo_section_candidates.append(lo_candidate_px)

                if len(hi_section_candidates) == section_w:
                    bad_counter = bad_counter + 1
                    # mark the overlay frame with the depth calculation lines
                    for px in hi_section_candidates:
                        width_calc_overlay_im[px[0], px[1]] = WHITE
                    for px in lo_section_candidates:
                        width_calc_overlay_im[px[0], px[1]] = WHITE
                    # Set result variables
                    hi_ave_y = np.mean(hi_section_candidates, axis=0)[get_y]
                    lo_ave_y = np.mean(lo_section_candidates, axis=0)[get_y]
                    ave_px_width = lo_ave_y - hi_ave_y
                    width, depth = self.calc_w_and_d(ave_px_width, color_frame, raw_filtered_depth, range_start, range_end, hi_ave_y, lo_ave_y)
                    comp_width, null = self.calc_w_and_d(ave_px_width, color_frame, raw_filtered_depth, range_start, range_end, hi_ave_y, lo_ave_y, fixed_depth=True)
                    sections_found.append([range_start, range_end, width, depth, ave_px_width, comp_width])
                    hi_section_candidates = []
                    lo_section_candidates = []
                    range_end = curr_x_idx - 1
                    range_start = range_end - section_w + 1
                    curr_x_idx = range_end
                else:
                    curr_x_idx = curr_x_idx - 1     

        if len(sections_found) < 4:
            return False, None, None, None, None

        # Get the height above the filament:
        ave_results = np.mean(sections_found, axis=0)
        ave_width = ave_results[2]
        ave_depth = ave_results[3]
        most_recent_width = sections_found[0][2]
        most_recent_comp_width = sections_found[0][5]
        most_recent_depth = sections_found[0][3]
        # print(sections_found)
        width_list_cm = [k[2] for k in sections_found] # sections_found[:,2]
        depth_list_cm = [k[3] for k in sections_found] # sections_found[:,3]
        center_x_sections = np.array(np.array([k[3] for k in sections_found]) + int(np.trunc(section_w/2)))

        if speed != None:       # Calculate an estimate for the change in width 
            # Do a linear regression to get the changes over time
            dist_traveled_in_px = np.abs(center_x_sections[0] - center_x_sections[-1])
            dist_traveled_in_cm = (dist_traveled_in_px/x_size) * 2 * ave_depth * np.tan(self.FOV_H_RAD/2)
            
            center_x_sections_cm = center_x_sections * 2 * ave_depth * np.tan(self.FOV_H_RAD/2)
            slope, offset = np.polyfit(center_x_sections_cm, width_list_cm, deg=1)

            width_slope = slope * speed     # (cm/cm)*(cm/s) = cm/s (change in width per time)            

            ### GRAPHS UPDATE ###
            self.add_graph_data(data=most_recent_depth, extra1=most_recent_width, extra2=most_recent_comp_width, type="fil_depth")
            ### GRAPHS UPDATE ###

            return True, most_recent_width, most_recent_depth, width_calc_overlay_im, width_slope
        else:

            ### GRAPHS UPDATE ###
            self.add_graph_data(data=most_recent_depth, extra1=most_recent_width, extra2=most_recent_comp_width, type="fil_depth")
            ### GRAPHS UPDATE ###
            return True, most_recent_width, most_recent_depth, width_calc_overlay_im, None
        

    def calc_w_and_d(self, ave_px_width, proc_color, proc_depth_frame, x_left, x_right, y_up, y_down, fixed_depth=False):
        """
        Arguments are inclusive range
        Probe the depth FOV in a few places in range to get an average depth map. 
        Using average depth and the pixel width, calculate an approximate width of
        the filament. 
        """
        
        # Probe the depth of the pixels
        num_x_probes = 2
        num_y_probes = 5
        range_width = x_right - x_left
        range_height = y_down - y_up
        x_spacing = range_width / (num_x_probes + 1)
        y_spacing = range_height / (num_y_probes + 1)
        im_shape = proc_color.shape
        height_px_size = im_shape[0]
        width_px_size = im_shape[1]
        depth_calibration_val = -4

        # For each px, get the depth at that point and store the px. 
        probe_pxs = []
        depth_readings = []
        for i in range(num_x_probes):
            for k in range(num_y_probes):
                x_coord = int(np.round(((i * x_spacing + 1) + x_left),0))
                y_coord = int(np.round(((k * y_spacing + 1) + y_up),0))
                probe_pxs.append(np.array([y_coord, x_coord]))
                depth_readings.append(proc_depth_frame[y_coord, x_coord]/10)

        if fixed_depth:
            average_depth = 75  # (mm) should always be the same as target depth value
        else:
            # Measured depths from the depth camera are planar depth not radial depth
            average_depth = np.round(sum(depth_readings)/len(depth_readings), 3)
            average_depth = average_depth + depth_calibration_val
        calc_width = (ave_px_width/height_px_size) * 2 * average_depth * np.tan(self.FOV_V_RAD/2)

        depth_real = average_depth
        width_real = calc_width
        return width_real, depth_real

    def collate(self, frame_1, frame_2, color_frame=None):
        """
        Returns 2 frame: 
        (second frame only returned if color_frame is provided)
            First frame: Overlaying the detected elements from depth and color frames
                Red = Just white pixels from `frame_1`
                Green = Just white pixels from `frame_2`
                Blue = White pixels from both frames
                Black = Black (or non-white) pixel from both frames
            Second frame: Same thing but replace the black part of the image with the
                          original color image
        """
        BLUE =  (255,0,0)
        GREEN = (0,255,0)
        RED =   (0,0,255)
        BLACK = (0,0,0)
        WHITE = (255,255,255)

        shape = frame_1.shape
        if shape != frame_2.shape:
            raise RuntimeError("Input frames must be same shape for collation")
        if color_frame is not None:
            if frame_1.shape != shape:
                raise RuntimeError("Color frame must be same shape as input frames")

        # Create a new image which is an overlay of the two (on black background)
        collate = np.zeros((shape[0], shape[1], 3))
        collate[np.logical_and(frame_1, frame_2)]                                      = BLUE      # BOTH
        collate[np.logical_and(np.logical_not(frame_1), frame_2)]                      = RED       # JUST 1
        collate[np.logical_and(frame_1, np.logical_not(frame_2))]                      = GREEN     # JUST 2
        collate[np.logical_and(np.logical_not(frame_1), np.logical_not(frame_2))]      = BLACK     # NEITHER

        final_on_black = collate # gray_thresh_frame

        # Put  onto the original color frame as an overlay
        blue_mask = np.all(collate == BLUE, axis=-1)
        green_mask = np.all(collate == GREEN, axis=-1)
        red_mask = np.all(collate == RED, axis=-1)

        if color_frame is not None:
            color_frame = color_frame.__deepcopy__(color_frame)
            mod_color_frame = color_frame
            mod_color_frame[blue_mask] = BLUE
            mod_color_frame[green_mask] = GREEN
            mod_color_frame[red_mask] = RED

            final_overlayed = mod_color_frame
            return final_on_black, final_overlayed
        else:
            return final_on_black, None

    def process_frame(self, color_frame, colorized_depth_frame, raw_depth_frame):
        """
        Performs the image processing on a single frame to determine filaments
        Returns: (processed_color_frame, processed_depth_frame)
        """

        # Reducing magic numbers:
        WHITE = 255
        
        color_blurred_frame1 = color_frame
        color_blurred_frame1 = self.color_remove_white(color_frame=color_blurred_frame1)

        # removes individual pixel noise with median filter:
        color_blurred_frame1 = cv2.medianBlur(src=color_blurred_frame1, ksize=3)

        # Start by blurring the frame but maintaining edges:

        color_blurred_frame1 = cv2.GaussianBlur(src=color_blurred_frame1, ksize=(3,3), sigmaX=0.5, sigmaY=0.5)

        # Convert to grayscale
        gray_blurred1 = cv2.cvtColor(color_blurred_frame1, cv2.COLOR_BGR2GRAY)

        gray_blurred1 = cv2.GaussianBlur(src=gray_blurred1, ksize=(5,5), sigmaX=0.5, sigmaY=0.5)

        # removes individual pixel noise with 5 iterations of median filter
        gray_blurred1 = cv2.medianBlur(src=gray_blurred1, ksize=3)
        gray_blurred1 = cv2.medianBlur(src=gray_blurred1, ksize=3)
        gray_blurred1 = cv2.medianBlur(src=gray_blurred1, ksize=3)
        gray_blurred1 = cv2.medianBlur(src=gray_blurred1, ksize=3)
        gray_blurred1 = cv2.medianBlur(src=gray_blurred1, ksize=3)

        # Adaptive threshold:
        gray_thresh_frame1 = cv2.adaptiveThreshold(gray_blurred1,WHITE,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,7,3)
        kernel = np.ones((3,3),np.uint8)    # chooses area and weights over which to pass
        gray_thresh_frame1 = cv2.morphologyEx(gray_thresh_frame1, cv2.MORPH_OPEN, kernel, iterations=1)
        gray_thresh_frame1 = cv2.morphologyEx(gray_thresh_frame1, cv2.MORPH_ERODE, kernel, iterations=1)

        # Depth frame processing
        # Use a Gaussian Blur and Median Filter to simply reduce the noise
        raw_depth_median = cv2.medianBlur(src=raw_depth_frame, ksize=3)
        raw_depth_blur = cv2.GaussianBlur(raw_depth_median, ksize=(5,5), sigmaX=100, sigmaY=100)
        
        final_color_frame = gray_thresh_frame1 
        final_colorized_depth_frame = colorized_depth_frame
        final_raw_depth_data =  raw_depth_blur

        return final_color_frame, final_colorized_depth_frame, final_raw_depth_data

    def color_remove_white(self, color_frame):
        """
        Function takes in a color frame and extracts the white pixels and then 
        sets them to match the surrounding image
        """
        w_thresh = 240      # Threshold value for where all values are greater than this value, we count as white
        WHITE = 255
        diff_thresh = WHITE * 0.05 * 3

        cf = color_frame.__deepcopy__(color_frame)
        shape = cf.shape
        y_size = shape[0]
        x_size = shape[1]
        
        gray_color = cv2.cvtColor(cf, cv2.COLOR_BGR2GRAY)
        ret, thresh_frame = cv2.threshold(src=gray_color, thresh=w_thresh, maxval=WHITE, type=cv2.THRESH_BINARY)

        # create markers from all of the white sections
        success, markers = cv2.connectedComponents(thresh_frame.astype(np.uint8))
        num_markers = len(np.unique(markers))

        new_im = np.zeros((shape[0], shape[1], 1))
        new_im[markers == 0] = 255

        # for each unique mark, set the value of all of the pixels in this section to the average of just beyond the extreme pixels
        for mrk in range(num_markers):
            if mrk == 0:
                continue # don't count the background (this is necessary)
            roi = np.argwhere(markers == mrk)   # region of interest
            roi = [k for k in roi]
            roi.sort(key=self.__sort_by_x__)
            x_min = roi[0]
            x_max = roi[-1]
            roi.sort(key=self.__sort_by_y__)
            y_min = roi[0]
            y_max = roi[-1]

            extreme_px = []
            try:
                px_val_x_max = color_frame[x_max[0], x_max[1]+1]
                extreme_px.append(px_val_x_max)
            except:
                pass
            try:
                px_val_x_min = color_frame[x_min[0], x_min[1]-1]
                extreme_px.append(px_val_x_min)
            except:
                pass
            try:
                px_val_y_max = color_frame[y_max[0]+1, y_max[1]]
                extreme_px.append(px_val_y_max)
            except:
                pass
            try:
                px_val_y_min = color_frame[y_min[0]-1, y_min[1]]
                extreme_px.append(px_val_y_min)
            except:
                pass

            mean_val = np.mean(extreme_px, axis=0)
            median_val = np.median(extreme_px, axis=0)
            if np.sum(np.abs(mean_val - median_val)) > diff_thresh:
                result_val = (median_val[0].astype(np.uint8),median_val[1].astype(np.uint8),median_val[2].astype(np.uint8))
            else:
                result_val = (mean_val[0].astype(np.uint8),mean_val[1].astype(np.uint8),mean_val[2].astype(np.uint8))

            # Set picture frame to calculated values
            cf[markers==mrk] = result_val

        corrected_frame = cf

        return corrected_frame
    

    def compare_depth_to_color(self, color_frame, raw_depth_frame):
        """
        Returns 2 frames: notep
            First frame: Overlaying the detected elements from depth and color frames
                Red = Just depth frame
                Green = Just color frame
                Blue = Both frames 
                Black = Neither frame
            Second frame: Same thing but replace the black part of the image with the
                          original color image
        """

        # Reducing magic numbers:
        WHITE = 255
        BLACK = 0
        BLUE = [255,0,0]
        GREEN = [0,255,0]
        RED = [0,0,255]

        # DEPTH FRAME PROCESSING

        gray_depth = raw_depth_frame

        max_val = 1000  # np.max(gray_depth)
        min_val = 300   # np.min(gray_depth)

        # Shift the minimum value to better separate surroundings
        gray_depth = gray_depth - min_val
        gray_depth[gray_depth < 0] = 0
        
        # Shift the maximum value to better separate surroundings from focus 
        if max_val != 0:
            gray_depth = gray_depth / (max_val-min_val)
            gray_depth = gray_depth * 255  # shifts the maximum value to 255 (white)
        else:
            print("divide by 0")
        gray_depth = np.floor(gray_depth)           # rounds every digit to get to the nearest int
        gray_depth = gray_depth.astype(np.uint8)    # now even values should have no errors converting to ints

        # Inverse colors so that foreground is lighter than background:
        gray_depth = cv2.bitwise_not(gray_depth)
        gray_depth[gray_depth == max_val - min_val] = 255

        depth_thresh_frame = cv2.adaptiveThreshold(gray_depth,WHITE,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,7,2)

        # COLOR FRAME PROCESSING

        # Start by blurring the frame but maintaining edges:
        color_blurred_frame = self.bilateral_blur_frame(color_frame, 5, 150, 150)

        # Convert to grayscale
        gray_blurred = cv2.cvtColor(color_blurred_frame, cv2.COLOR_BGR2GRAY)

        # Use adaptive mean thresholding and an Opening operation to remove some noise
        gray_thresh_frame = cv2.adaptiveThreshold(gray_blurred,WHITE,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,7,4)
        kernel = np.ones((3,3),np.uint8)    # chooses area and weights over which to pass
        gray_thresh_frame = cv2.morphologyEx(gray_thresh_frame, cv2.MORPH_OPEN, kernel, iterations=1)

        # Create a new image which is an overlay of the two (on black background)
        collate = np.zeros((raw_depth_frame.shape[0], raw_depth_frame.shape[1], 3))
        collate[np.logical_and(gray_thresh_frame, depth_thresh_frame)]                                      = BLUE      # BOTH
        collate[np.logical_and(np.logical_not(gray_thresh_frame), depth_thresh_frame)]                      = RED       # JUST DEPTH
        collate[np.logical_and(gray_thresh_frame, np.logical_not(depth_thresh_frame))]                      = GREEN     # JUST COLOR
        collate[np.logical_and(np.logical_not(gray_thresh_frame), np.logical_not(depth_thresh_frame))]      = BLACK     # NEITHER

        final_on_black = collate # gray_thresh_frame

        # Put  onto the original color frame as an overlay
        blue_mask = np.all(collate == BLUE, axis=-1)
        green_mask = np.all(collate == GREEN, axis=-1)
        red_mask = np.all(collate == RED, axis=-1)

        mod_color_frame = color_frame
        mod_color_frame[blue_mask] = BLUE
        mod_color_frame[green_mask] = GREEN
        mod_color_frame[red_mask] = RED

        final_overlayed = mod_color_frame

        return final_on_black, final_overlayed
    

    def mouse_info(self, frame, frame_name, type="gray", depth_frame=None):

        type_list = ["gray", "depth", "color"]

        if type.lower() not in type_list:
            raise RuntimeError("Provided `type` of '{}' is not a legal type. Legal types are {}".format(type, type_list))
        
        if type.lower() == "depth" and depth_frame.any() == None:
            raise RuntimeError("Must provide `depth_data` when `type`='depth'")

        # Get coords of mouse
        cv2.setMouseCallback(frame_name, get_mouse_info)

        if type.lower() == "gray":
            gray_value = frame[mouse_point[1], mouse_point[0]]

            text = "I{}".format(gray_value)
            loc = (mouse_point[0] - 50, mouse_point[1] - 20)
            font = cv2.FONT_HERSHEY_DUPLEX
            color = (200, 200, 0)
            scale = 1
            thickness = 2
            cv2.putText(frame, text, loc, font, scale, color, thickness)
        elif type.lower() == "depth":
            depth_value = depth_frame[mouse_point[1], mouse_point[0]] / 100.0

            text = "D{}".format(depth_value)
            loc = (mouse_point[0] - 50, mouse_point[1] - 20)
            font = cv2.FONT_HERSHEY_DUPLEX
            color = (255, 255, 255)
            scale = 1
            thickness = 2
            cv2.putText(frame, text, loc, font, scale, color, thickness)
        elif type.lower() == "color":
            color_values = frame[mouse_point[1], mouse_point[0]]
            B = color_values[0]; G = color_values[1]; R = color_values[2]

            text = "BGR:({}, {}, {})".format(B, G, R)
            loc = (mouse_point[0] - 50, mouse_point[1] - 20)
            font = cv2.FONT_HERSHEY_DUPLEX
            color = (200, 200, 0)
            scale = 1
            thickness = 2
            cv2.putText(frame, text, loc, font, scale, color, thickness)
        else:
            print("Wrong type")


    def process_video(self, video_name:str, mouse_info=True, window_name="Color"):
        """
        Processes all of the frames of a video

        Parameters:
            video_name : str
                path to the video + the video name - _type - extension
            mouse_info : bool
                should the mouse add additional info to the video overlay
            window_name : str
                what the name of the resulting window the video will play on
        """
        

        color_video_name = video_name + "_color.avi"
        depth_video_name = video_name + "_depth.avi"
        depth_data_file_name = video_name + "_depth.pkl"
        color_cap = cv2.VideoCapture(color_video_name)
        depth_cap = cv2.VideoCapture(depth_video_name)
        with open(depth_data_file_name, "rb") as fp:   # Unpickling
            depth_data = pickle.load(fp)
        total_frames = color_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        advance_frames=True
        reverse = False

        # Collect RGB Frame information in advance to process simultaneously
        color_frames = []
        counter = 0
        while color_cap.isOpened():
            ret, color_frame = color_cap.read()
            if ret == False:
                raise RuntimeError("Can't read frames for some reason. Exiting")
            color_frames.append(color_frame)

            if counter == len(depth_data)-1:
                break

            counter = counter + 1
        color_cap.release()

        # Collect Colorized Depth Frame information in advance to process simultaneously
        colorized_depth_frames = []
        counter = 0
        while depth_cap.isOpened():
            ret, depth_frame = depth_cap.read()
            if ret == False:
                raise RuntimeError("Can't read frames for some reason. Exiting")
            colorized_depth_frames.append(depth_frame)
            color_frames.append(color_frame)

            if counter == len(depth_data)-1:
                break

            counter = counter + 1
        depth_cap.release()

        blank_img = np.zeros((color_frame.shape[0], color_frame.shape[1]))    # purely for reference for collating

        # Iterate through all frames with total data
        idx = 0
        time_stamp = time.time()
        successful_cnt = 0
        tot_cnt = 0 
        while idx < total_frames:
            print("Time for iteration = {} sec".format(np.round(time.time() - time_stamp, 4)))
            time_stamp = time.time()
            # Collect frame data in 1 place
            curr_frames = (color_frames[idx].__deepcopy__(color_frames[idx]), 
                           colorized_depth_frames[idx].__deepcopy__(colorized_depth_frames[idx]), 
                           depth_data[idx].__deepcopy__(depth_data[idx])) 

            # For final product video
            processed_curr_frames = self.process_frame(color_frame=curr_frames[0],
                                                       colorized_depth_frame=curr_frames[1],
                                                       raw_depth_frame=curr_frames[2])
            outlined_img = self.outline_filament(processed_curr_frames[0])
            ret, width, depth, width_im, width_slope = self.calc_filament_props(outlined_frame=outlined_img, color_frame=curr_frames[0], 
                                                                                raw_filtered_depth=processed_curr_frames[2], speed=None)
            
            if ret:
                print("Successful Depth Calc: width = {}; depth = {}".format(np.round(width, 3), np.round(depth, 3)))
                on_black, on_color = self.collate(width_im, blank_img, color_frame=curr_frames[0])
                depth_frame_2_show = on_color # width_im # outlined_img # width_im
                color_frame_2_show = outlined_img # curr_frames[0]
                successful_cnt = successful_cnt + 1
                tot_cnt = tot_cnt + 1
            else:
                print("Failed Depth Calc")
                depth_frame_2_show = outlined_img
                color_frame_2_show = curr_frames[0]
                tot_cnt = tot_cnt + 1

            
            

            # # To get comparison video
            color_window_name = window_name + " Color"
            depth_window_name = window_name + " Depth"
            cv2.namedWindow(color_window_name)
            cv2.namedWindow(depth_window_name)

            if mouse_info:
                self.mouse_info(frame=color_frame_2_show, frame_name=color_window_name, type="gray", depth_frame=curr_frames[2])
                self.mouse_info(frame=depth_frame_2_show, frame_name=depth_window_name, type="depth", depth_frame=curr_frames[2])
            
            cv2.imshow(color_window_name, color_frame_2_show)
            cv2.imshow(depth_window_name, depth_frame_2_show)
            
            key_press = cv2.waitKey(33)
            if key_press == ord('q'):
                success_rate = np.round(successful_cnt/tot_cnt, 3) * 100
                print("Successful frames = {} from {} Total frames; Success Rate = {}%".format(successful_cnt, tot_cnt, success_rate))
                break
            elif key_press == ord('p'):
                advance_frames = not(advance_frames)
            elif key_press == ord('r'):
                reverse = not(reverse)

            # If we are supposed to be iterating through frames, then advance
            if advance_frames == True:
                if not(reverse):
                    idx = idx + 1
                else:
                    if idx != 0:
                        idx = idx - 1


    def play_video(self, video_name:str, type:str, mouse_info = False):
        """
        Plays video from the filepath\\name `video_name`
        """
        type = type.lower()
        cap = cv2.VideoCapture(video_name)
        if type == "depth":
            if "_depth" not in video_name:
                raise RuntimeError("Video name should include depth in order to be `type`='depth'")
            depth_data_file_name = video_name[:-4] + ".pkl"
          # should import all images in shape [height, width, num_images]
            with open(depth_data_file_name, "rb") as fp:   # Unpickling
                depth_data = pickle.load(fp)
        else:
            depth_data = None
        advance_frames=False
        last_frame = None
        counter = 0 

        while cap.isOpened():
            if advance_frames == True:
                ret, frame = cap.read()

                ### DEPTH IMAGE PLOTTING ###
                mean, stddev, max, min = self.calc_depth_frame_info(depth_data[counter])
                self.add_graph_data(mean, stddev, max, min, type="ave_depth")
                ### DEPTH IMAGE PLOTTING END ###

                if type.lower() == "gray":
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                if counter == 0:
                    ret, frame = cap.read()

                    if type.lower() == "gray":
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    counter = counter + 1
                else:
                    frame = last_frame
                    ret = True
            last_frame = frame.copy()
    
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            window_name = video_name[:-4]
            cv2.namedWindow(window_name)

            # writes to frame data about the mouse position
            if mouse_info:
                if depth_data:
                    self.mouse_info(frame, window_name, type=type, depth_frame=depth_data[counter])
                else:
                    self.mouse_info(frame, window_name, type=type)

            cv2.imshow(window_name, frame)
            key_press = cv2.waitKey(33)
            if key_press == ord('q'):
                break
            elif key_press == ord('p'):
                advance_frames = not(advance_frames)

            if counter == len(depth_data)-1:
                break
            
            if advance_frames:
                counter = counter + 1

        cap.release()

    def calc_depth_frame_info(self, raw_depth_frame):
        """
        filters out all of the pixels that are reading 0 and then take the relevant info
        returns mean, stddev, max, and min
        """
        filtered_depth = raw_depth_frame.flatten()
        filtered_depth = filtered_depth[filtered_depth > 0]
        mean = np.mean(filtered_depth)
        stddev = np.std(filtered_depth)
        filtered_depth = filtered_depth[filtered_depth < mean + (0.01*stddev)] #np.array([k for k in filtered_depth if k > 0])
        mean = np.mean(filtered_depth)
        stddev = np.std(filtered_depth)
        max = filtered_depth.max()
        min = filtered_depth.min()

        return mean, stddev, max, min


    def add_graph_data(self, data, error_bar=None, extra1=None, extra2=None, type="ave_depth"):
        """
        Adds data to be able to make plots automatically
        """
        curr_time = np.round(time.time() - self.time_zero, 4)
        type = type.lower()
        if type == "ave_depth":
            """
            error_bar values are 1 stddev
            extra1 is the maximum recorded value
            extra2 is the minimum recorded value
            """
            self.depth_graph_data.append(np.round(data, 4))
            self.depth_graph_err_data.append(np.round(error_bar, 4))
            self.depth_graph_max_data.append(np.round(extra1, 4))
            self.depth_graph_min_data.append(np.round(extra2, 4))
            self.depth_independent_data.append(curr_time)

        elif type == "depth_control":
            self.depth_control_graph_data.append(data)
            self.depth_independent_data.append(curr_time)

        elif type == "fil_depth":
            """
            extra1 = width data
            extra2 = width data calculation at constant depth value instead of measured
            """
            self.fil_depth_graph_data.append(np.round(data, 4))
            self.fil_width_graph_data.append(np.round(extra1, 4))
            self.fil_width_compare_data.append(np.round(extra2, 4))
            self.fil_depth_independent_data.append(curr_time)

        else:
            raise RuntimeError("Illegal plot type provided for graph data.")

        return 
    
    def create_graphs(self):
        """
        Creates plots from all of the data generated
        """
        curr_time = time.time()
        if len(self.depth_graph_data) > 0:
            self.depth_independent_data = np.array(self.depth_independent_data)
            normalized_x = (self.depth_independent_data-self.depth_independent_data[0])
            normalized_x = normalized_x*5.0/(normalized_x.max())
            ax = plt.figure()
            plt.title("2 $\sigma$ Filtered Average Values of Flat Depth Video no Light for 75 mm Gap")    
            plt.errorbar(normalized_x, self.depth_graph_data, yerr=self.depth_graph_err_data)
            plt.xlabel("Time (s)")    
            plt.ylabel("Average Measured Depth Value (mm)")    
            plt.savefig("graph_outputs\\depth_plot_{}.png".format(int(curr_time)), format='png', dpi=300)   

            ax = plt.figure()
            plt.title("0.01 $\sigma$ Filtered Extreme Values of Flat Depth Video no Light for 75 mm Gap")    
            plt.plot(normalized_x, self.depth_graph_max_data)
            plt.plot(normalized_x, self.depth_graph_data)
            plt.plot(normalized_x, self.depth_graph_min_data)
            plt.xlabel("Time (s)")    
            plt.ylabel("Measured Depth Value (mm)")    
            plt.legend(["Max", "Mean", "Min"])
            plt.savefig("graph_outputs\\depth_plot_extremes_{}.png".format(int(curr_time)), format='png', dpi=300)    

        if len(self.depth_control_graph_data) > 0:
            ax = plt.figure()
            plt.title("Depth Control Progression")    
            plt.errorbar(self.depth_control_independent_data, self.depth_control_graph_data, y_err=self.depth_control_graph_err_data)
            plt.xlabel("Time (s)")    
            plt.ylabel("Average Measured Depth Value of filament (mm)")    
            plt.savefig("graph_outputs\\depth_control_plot_{}.png".format(int(curr_time)), format='png', dpi=300)  

        if len(self.fil_depth_graph_data) > 0:
            self.fil_depth_independent_data = np.array(self.fil_depth_independent_data)
            normalized_x = (self.fil_depth_independent_data-self.fil_depth_independent_data[0])
            normalized_x = normalized_x*5.0/(normalized_x.max())
            
            # Shifts average to 0
            d_overlay = self.rec_data.known_12mm_fil_depth - np.mean(self.rec_data.known_12mm_fil_depth)
            # flips data about y axis because height of filament and depth to filament are inversely related
            d_overlay = -1.0*d_overlay
            # shift up by mean of the depth to show height variation over depth
            d_overlay = d_overlay + np.mean(self.fil_depth_graph_data) 
            d_measure_max = d_overlay.max() *np.ones(normalized_x.shape)
            d_measure_min = d_overlay.min() *np.ones(normalized_x.shape)
            d_measure_mean = np.mean(d_overlay) *np.ones(normalized_x.shape)
            d_measure_target = 75.0 *np.ones(normalized_x.shape)
            d_overlay_normalized = np.array(range(0,16)) /3.0
            ax = plt.figure()
            plt.title("Moving Known Filament Depth Measurement for 75 mm Gap")    
            plt.plot(normalized_x, self.fil_depth_graph_data)
            plt.plot(d_overlay_normalized, d_overlay, linestyle="dashed")
            plt.plot(normalized_x, d_measure_target, linestyle="dotted")
            plt.plot(normalized_x, d_measure_max, linestyle="dotted")
            plt.plot(normalized_x, d_measure_mean, linestyle="dotted")
            plt.plot(normalized_x, d_measure_min, linestyle="dotted")
            plt.xlabel("Time (s)")    
            plt.ylabel("Measured Depth Value (mm)")    
            plt.legend(["Camera", "Actual", "Target", "Max", "Mean", "Min"])
            plt.savefig("graph_outputs\\fil_depth_plot_{}.png".format(int(curr_time)), format='png', dpi=300)  

            w_measure_max = self.rec_data.known_12mm_fil_width.max() *np.ones(normalized_x.shape)
            w_measure_min = self.rec_data.known_12mm_fil_width.min() *np.ones(normalized_x.shape)
            w_measure_mean = np.mean(self.rec_data.known_12mm_fil_width) *np.ones(normalized_x.shape)
            w_measure_target = 11.60 *np.ones(normalized_x.shape)
            ax = plt.figure()
            plt.title("Moving Known Filament Width Measurement for 75 mm Gap")    
            plt.plot(normalized_x, self.fil_width_graph_data)
            plt.plot(normalized_x, self.fil_width_compare_data)
            plt.plot(normalized_x, w_measure_target, linestyle="dotted")
            plt.plot(normalized_x, w_measure_max, linestyle="dashed")
            plt.plot(normalized_x, w_measure_mean, linestyle="dashed")
            plt.plot(normalized_x, w_measure_min, linestyle="dashed")
            plt.xlabel("Time (s)")    
            plt.ylabel("Measured Width Value (mm)")    
            plt.legend(["Calc Depth", "Const Depth","Target", "Max", "Mean", "Min"])
            plt.savefig("graph_outputs\\fil_width_plot_{}.png".format(int(curr_time)), format='png', dpi=300)   

def main():
    """
    Runs testing suite to verify function of the IntelD405 class
    """

    # cam = IntelD405()
    im = ImageModule()

    # while (True):
    #     cam.show_raws()

    #     # Hit q to quit.
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # # Process single frame:
    video_name = 'video_outputs\\PR2CamBoogaloo2mmOverlap'
    BEFORE_LIGHT_FRAME = 325
    IN_LIGHT_FRAME = 355
    frame_num = IN_LIGHT_FRAME

    cap_color = cv2.VideoCapture(video_name + '_color.avi')
    cap_depth = cv2.VideoCapture(video_name + '_depth.avi')
    depth_data = None
    total_frames = cap_color.get(cv2.CAP_PROP_FRAME_COUNT)
    cap_color.set(1, frame_num)
    cap_depth.set(1, frame_num)
    ret, color_frame = cap_color.read()
    ret, depth_frame = cap_depth.read()
    proc_color, proc_depth, proc_raw_depth = im.process_frame(color_frame=color_frame, colorized_depth_frame=depth_frame, raw_depth_frame=depth_data)
    proc_depth = im.outline_filament(proc_color_frame=proc_color)
    while True:
        cv2.imshow("Processed Color Frame", proc_color)
        cv2.imshow("Processed Depth Frame", proc_depth)#proc_depth)

        key_press = cv2.waitKey(33)
        if key_press == ord('q'):
            break
        if key_press == ord('a'):
            frame_num = frame_num + 1
            ret, color_frame = cap_color.read()
            ret, depth_frame = cap_depth.read()
            proc_color, proc_depth, proc_raw_depth = im.process_frame(color_frame=color_frame, colorized_depth_frame=depth_frame, raw_depth_frame=depth_data)
            proc_depth = im.outline_filament(proc_color_frame=proc_color)
        if key_press == ord('r'):
            frame_num = frame_num - 1
            cap_color.set(1, frame_num)
            cap_depth.set(1, frame_num)
            ret, color_frame = cap_color.read()
            ret, depth_frame = cap_depth.read()
            proc_color, proc_depth, proc_raw_depth = im.process_frame(color_frame=color_frame, colorized_depth_frame=depth_frame, raw_depth_frame=depth_data)
            proc_depth = im.outline_filament(proc_color_frame=proc_color)

    cv2.destroyAllWindows()


# Runs the main method upon execution of the file
if __name__ == '__main__': 
    main()