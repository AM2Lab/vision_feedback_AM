##############################################################
#
#  Modifies G-code files for using Feedback Control 
#  File input is a standard SCARA G-code 
#  Outputs discretized G-code to within <0.1s commands 
#  new axis A for the camera position around the extruder
#  Code Created by Benjamin Gorse for Princeton 2024 Sr Thesis
#
##############################################################

# NOTE: INPUT FILE MUST END WITH A BLANK NEWLINE AT EOF
# NOTE: ASSUMES SPEEDS ARE IN MM/MIN --> CHANGE SPEED_FACTOR TO 1 FOR MM/S

from typing import Dict, Tuple
import numpy as np
import os
import copy

# Constants set for code function *** NOT TO CHANGE ***
PATH_HERE = os.path.dirname(os.path.abspath(__file__))
PATH_DESKTOP = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
FLOAT_RND_ERR_THRESHOLD = 0.0001
RAD_2_DEG = 180.0/np.pi
DEG_2_RAD = np.pi/180.0
SCARA_ARM_L1 = 546.9        # (mm)
SCARA_ARM_L2 = 456.34       # (mm)
CAM_AXIS_NAME = "C"
AXES_LIST = ["X", "Y", "Z", "E", CAM_AXIS_NAME]   # List of strings representing axis names 
E_AXIS_DOMINANT = False     # If True, is incorporated into speed calculation. If False, then not     

# TO CHANGE: Variables to change in file main execution
PATH_EXPORT = os.path.join(PATH_HERE, 'gcode_outputs\\output_file_ex_name2.gcode') # path to export file to
PATH_READ = os.path.join(PATH_HERE, 'gcode_inputs\\input_file_ex_name.gcode') # path to read gcode from
DISC_T = 0.1                            # Threshold for the maximum time length of any specific command of Gcode
CAMERA_DISC_T = 0.075                   # Threshold for the maximum time length of G-code while camera makes a relatively large adjustment. 
FILE_START_MODE = "Absolute"            # Starting mode in case file lacks specification 
DEBUG = True                            # Print extra comments for debugging if True
TARGET_CAM_SPEED = 80                   # Speed of camera motor for most motions (deg/s)
MAX_CAM_SPEED = 100                     # Maximum speed for camera motor when required (deg/s)
PRINTER_COORDS_FROM_CENTER = (0.0, 0.0) # Gives the displacement from centered coords to system coords
ROUND_DECIMAL = 3                       # Number of places after the decimal point to include for rounded numbers
INIT_CAMERA_DIR = (0.0, -1.0)           # Vector in form (X, Y) for initial motion direction of the camera
NEW_FILE = True                         # Whether or not the input should create or modify a file
SPEED_FACTOR = 60.0                     # Conversion factor to calculate in mm/s 

def main(file_path: str = PATH_READ, new_file = True,
         output_path: str = PATH_EXPORT, disc_t: float = DISC_T,
         axes = AXES_LIST):
    """
    Description: 
        Automated system to create g-code for running feedback system on the
        SCARA printer. Opens the file along the path specified by file_path
        Discretizes all of the code segements to execute each command within
        a time <= disc_t. Adds "C" axis to set the position of the camera for 
        the feedback system. Outputs the modified file as output_path.gcode
        By default, takes inputs from gcode_inputs folder and delivers 
        output to gcode_outputs folder. 

    Parameters:
        file_path   : str
            String containing the file path to the file to modify.
        new_file    : bool
            True  -> preserves input file and creates new output
            False -> edits file in place
        output_path : str
            String containing the name to output the gcode file as.
        disc_t      : float (seconds)
            Value of the time threshold for which the discretized file 
            will guarentee that each command will execute within.  
        axes        : List[str]
            List of the axis names that could be contained in Gcode file
    """

    positions = {}
    for axis in axes:
        positions.update({axis:0.0})   # sets initial position of each axis to 0. 

    # discretize(file_path=file_path, new_file=new_file, output_path=output_path, 
    #            mode=FILE_START_MODE, disc_t=disc_t, positions=positions, debug=DEBUG)
    
    camera_and_disc(file_path=PATH_READ, new_file=new_file, output_path=PATH_EXPORT, 
               mode=FILE_START_MODE, disc_t=disc_t, positions=positions, debug=DEBUG, 
               init_motion_dir=INIT_CAMERA_DIR)


##############################################################
#                     PRIMARY FUNCTIONS                      #
##############################################################


def discretize(file_path: str = PATH_READ, new_file = True,
         output_path: str = PATH_EXPORT, disc_t: float = DISC_T, 
         mode: str="Absolute", positions: Dict=None, debug=False):
    """
    Description
        Helper function to discretize the file specified by file_path to gcode 
        commands which execute in no longer than disc_t seconds. 

    Parameters:
        file_path   : str
            String containing the file path to the file to modify.
        new_file    : bool
            True  -> preserves input file and creates new output
            False -> edits file in place
        output_path : str
            String containing the name to output the gcode file as.
        disc_t      : float (seconds)
            Value of the time threshold for which the discretized file 
            will guarentee that each command will execute within. 
        mode        : str ("Absolute" or "Relative")
            Determines with which convention the commands will be interpreted
            to start. If a command is passed to change convention, that will 
            be taken into account.  
        debug       : bool ()
            True  -> includes extra comments in output to clarify vs original file
            False -> includes only original file comments.  
    """

    # Prepares file for writing
    rd_file_ob = open(file_path, "r") # Accesses the file in read mode
    original_file_contents = rd_file_ob.readlines() # reads all lines and returns them as a list of strings
    rd_file_ob.close() # closes the file 
    if not new_file:
        # Erases old file to write in place
        open(file_path, "w").close() # erases file contents

        w_file_ob = open(file_path, "w") # open the file to write in it
    else:
        # Opens new file to write in a new file instead
        w_file_ob = open(output_path, "w") # open the file to write in it

    speed = None

    mv_cnt = 0
    ln_count = 1
    for line in original_file_contents:
        if debug:
            print("line={}; positions={}".format(ln_count, positions)) 
        prefix = get_prefix(line)
        if prefix == "G90": # Set to Absolute
            mode = "Absolute"
            write_ln = line
        elif prefix == "G91": # Set to Relative
            mode = "Relative"
            write_ln = line
        elif prefix == "G92": # Set axis position
            print("initial position before executing G92: positions={}".format(positions))
            cmd_info, comment = extract_G92info(line)
            # print("cmd_info = {}".format(cmd_info))
            if not cmd_info: # Checks if empty
                print("Runs set axis position on all axes")
                for axis_name in positions:
                    positions[axis_name] = 0.0
            else: 
                print("Runs set axis position on limited axes")
                for axis_name in cmd_info:
                    print("Setting \"{}\" axis position to {}".format(axis_name,cmd_info[axis_name]))
                    positions[axis_name] = cmd_info[axis_name]
            print("positions have been updated to positions={}".format(positions))
            write_ln = line
        elif prefix == "G0" or prefix ==  "G1":
            if mv_cnt == 0:
                w_file_ob.write(line) # Do not discretize the first line
                last_cmd_info, comment = extract_G1info(line)
                update_pos(cmd_info=last_cmd_info, positions=positions, real_positions=positions, mode=mode) # real_positions is not relevant here
                mv_cnt = mv_cnt + 1
                ln_count += 1
                continue
            
            cmd_info, comment = extract_G1info(line)

            axes_displacements = {}
            match_last_cmd = True
            tot_dist_sq = 0
            for axis_name in cmd_info:
                if axis_name == "F":
                    speed = cmd_info[axis_name]
                    continue
                if mode.lower() == "absolute":
                    axis_displacement = (cmd_info[axis_name] - positions[axis_name])
                    tot_dist_sq = tot_dist_sq + ((axis_displacement)**2)
                    axes_displacements.update({axis_name:axis_displacement})
                    if np.abs(axis_displacement) > FLOAT_RND_ERR_THRESHOLD: 
                        match_last_cmd = False
                elif mode.lower() == "relative":
                    axis_displacement = cmd_info[axis_name]
                    tot_dist_sq = tot_dist_sq + (cmd_info[axis_name]**2)
                    axes_displacements.update({axis_name:axis_displacement})
                    if np.abs(cmd_info[axis_name]) > FLOAT_RND_ERR_THRESHOLD:
                        match_last_cmd = False
                else:
                    raise RuntimeError("mode set incorrectly to \"{}\"".format(mode))
                
            if match_last_cmd == True:
                last_cmd_info = cmd_info
                w_file_ob.write(line)
                ln_count += 1
                continue        # No need to discretize if it barely / does not move

            tot_dist = np.sqrt(tot_dist_sq)
            try:
                tot_time = tot_dist / speed
            except:
                if speed == 0:
                    print("speed is 0")
                else:
                    print("speed is None")
            
            disc_parts = int(np.ceil(tot_time / disc_t)) 

            if disc_parts <= 1:     # No need to discretize if it is already within threshold
                w_file_ob.write(line)
                continue
            for disc_idx in range(disc_parts):
                result_ln = prefix + " "
                for axis_name in positions:
                    if axis_name in axes_displacements:
                        if mode.lower() == "absolute":
                            disc_displacement = np.round(axes_displacements[axis_name]/float(disc_parts), 3) 
                            result_ln = result_ln + axis_name + str(np.round(disc_displacement + positions[axis_name],3)) + " "
                            positions[axis_name] = positions[axis_name] + disc_displacement
                        elif mode.lower() == "relative":
                            disc_displacement = np.round(axes_displacements[axis_name]/disc_parts, 3)
                            result_ln = result_ln + axis_name + str(np.round(disc_displacement,3)) + " "
                        else:
                            raise RuntimeError("mode set incorrectly to \"{}\"".format(mode))
                    else:
                        if mode.lower() == "absolute":
                            result_ln = result_ln + axis_name + str(np.round(positions[axis_name],3)) + " "
                        elif mode.lower() == "relative":
                            result_ln = result_ln + axis_name + str(0.0) + " "
                        else:
                            raise RuntimeError("mode set incorrectly to \"{}\"".format(mode))

                result_ln = result_ln + "F" + str(speed) + " "
                if disc_idx == 0 and comment:
                    result_ln = result_ln + comment
                if disc_idx == 0 and debug: 
                    result_ln = result_ln + "\n"
                    if comment:
                        subline = line.replace(comment, "")
                    else:
                        subline = line[:-1]
                    w_file_ob.write("; Discretization for: \"{}\" START \\/ [\n".format(subline))
                elif disc_idx == disc_parts - 1 and debug:
                    w_file_ob.write(result_ln + "\n")
                    if comment:
                        subline = line.replace(comment, "")
                    else:
                        subline = line[:-1]
                    result_ln = "; Discretization for: \"{}\" END   /\\ ]\n".format(subline)
                else: 
                    result_ln = result_ln + "\n"
                w_file_ob.write(result_ln)
                
            write_ln = None
            last_cmd_info = cmd_info
            for axis_name in positions:
                if axis_name not in last_cmd_info:
                    last_cmd_info.update({axis_name:positions[axis_name]})
            mv_cnt = mv_cnt + 1
        else:
            write_ln = line

        ln_count += 1

        if write_ln:    # Checks if write_ln != None
            w_file_ob.write(write_ln)

    w_file_ob.close()


def camera_and_disc(file_path: str = PATH_READ, new_file = True,
         output_path: str = PATH_EXPORT, mode: str="Absolute", 
         positions: Dict=None, debug=False, disc_t: float = DISC_T, 
         init_motion_dir: Tuple=(1.0,0.0), camera_dominant=False):
    """
    Description
        Helper function to add the camera movements to the Gcode in
        order to use feedback. Camera rotates between -180 deg <->
        180 deg with 0 pointing in the positive X direction. 

    Parameters:
        file_path   : str
            String containing the file path to the file to modify.
        new_file    : bool
            True  -> preserves input file and creates new output
            False -> edits file in place
        output_path : str
            String containing the name to output the gcode file as.
        init_motion_dir     : (float, float, float)
            Vector in the form (X, Y) for the direction direction of motion
            SCARA nozzle AFTER moving to its initial position
        mode        : str ("Absolute" or "Relative")
            Determines with which convention the commands will be interpreted
            to start. If a command is passed to change convention, that will 
            be taken into account.  
        debug       : bool 
            True  -> includes extra comments in output to clarify vs original file
            False -> includes only original file comments.  
        disc_t      : float (seconds)
            Value of the time threshold for which the discretized file 
            will guarentee that each command will execute within.   
    """

    # Prepares file for writing
    rd_file_ob = open(file_path, "r") # Accesses the file in read mode
    original_file_contents = rd_file_ob.readlines() # reads all lines and returns them as a list of strings
    rd_file_ob.close() # closes the file 
    if not new_file:
        # Erases old file to write in place
        open(file_path, "w").close() # erases file contents

        w_file_ob = open(file_path, "w") # open the file to write in it
    else:
        # Opens new file to write in a new file instead
        w_file_ob = open(output_path, "w") # open the file to write in it

    speed = None

    # Loop to go through and resolve all commands
    real_positions = copy.deepcopy(positions) # makes a deep copy where changes are not reflected in original positions
    mv_cnt = 0
    for line in original_file_contents:

        ### Might want to add a check to force next command to rotate around instead. 

        prefix = get_prefix(line)
        if prefix == "G90": # Set to Absolute
            mode = "Absolute"
            write_ln = line
        elif prefix == "G91": # Set to Relative
            mode = "Relative"
            write_ln = line
        elif prefix == "G92": # Set axis position
            cmd_info, comment = extract_G92info(line)
            if not cmd_info: # Checks if empty
                for axis_name in positions:
                    positions[axis_name] = 0.0      # changes positions but not real_positions

            else: 
                for axis_name in cmd_info:
                    positions[axis_name] = cmd_info[axis_name]
            if debug:
                print("Set axis positions to positions={}".format(positions))
            write_ln = line
        elif prefix == "G0" or prefix ==  "G1":
            cmd_info, comment = extract_G1info(line)

            # Resolves case in which this is the first movement <-- should not discretize first movement
            if mv_cnt == 0:
                condensed_ln = prefix + " "
                for axis_name in cmd_info:
                    if axis_name == "F":
                        targ_cam_ang = calc_camera_pos(mode="posdir", target_pos=(cmd_info["X"], cmd_info["Y"]), dir=init_motion_dir, units="deg") # in degrees
                        condensed_ln = condensed_ln + CAM_AXIS_NAME + str(np.round(targ_cam_ang,3)) + " "
                        speed = cmd_info["F"]/SPEED_FACTOR
                    condensed_ln = condensed_ln + axis_name + str(cmd_info[axis_name]) + " "
                if comment:
                    condensed_ln = condensed_ln[:-1] + comment # removes the last space
                else:
                    condensed_ln = condensed_ln # removes the last space
                w_file_ob.write(condensed_ln+"\n") # Do not discretize the first line
                mv_cnt = mv_cnt + 1
                update_pos(cmd_info=cmd_info, positions=positions, real_positions=real_positions, mode=mode)
                positions[CAM_AXIS_NAME] = targ_cam_ang     # this is not updated from the cmd_info
                real_positions[CAM_AXIS_NAME] = targ_cam_ang
                last_cmd_info = cmd_info
                continue

            # Calculates all axis displacements that occur throughout the WHOLE original command execution
            # If Camera axis already is set, adjusts it to account for X/Y position change
            axes_displacements = {}
            match_last_cmd = True
            xyz_dist_sq = 0
            for axis_name in cmd_info:
                if axis_name == "F":
                    speed = cmd_info[axis_name]/SPEED_FACTOR
                    continue
                if mode.lower() == "absolute":
                    if not(axis_name == CAM_AXIS_NAME):         # Camera position requires calculating the rotation due to change in X and Y
                        axis_displacement = (cmd_info[axis_name] - positions[axis_name])
                        if not(axis_name == "E"):
                            xyz_dist_sq = xyz_dist_sq + ((axis_displacement)**2)                           
                        axes_displacements.update({axis_name:axis_displacement})
                        if np.abs(axis_displacement) > FLOAT_RND_ERR_THRESHOLD: 
                            match_last_cmd = False
                elif mode.lower() == "relative":
                    if not(axis_name == CAM_AXIS_NAME):         # Camera position requires calculating the rotation due to change in X and Y
                        axis_displacement = (cmd_info[axis_name])
                        if not(axis_name == "E"):
                            xyz_dist_sq = xyz_dist_sq + (cmd_info[axis_name]**2)
                        axes_displacements.update({axis_name:axis_displacement})
                        if np.abs(cmd_info[axis_name]) > FLOAT_RND_ERR_THRESHOLD:
                            match_last_cmd = False
                else:
                    raise RuntimeError("mode set incorrectly to \"{}\"".format(mode))
                
            if match_last_cmd == True:
                if debug == True:
                    print("Matching last command!")
                update_pos(cmd_info=cmd_info, positions=positions, real_positions=real_positions, mode=mode)
                last_cmd_info = cmd_info
                mv_cnt = mv_cnt + 1
                w_file_ob.write(line)   # Assume camera is required to move neglibly so do not need to add it
                continue                # No need to discretize if it barely / does not move

            
            # Net position that the camera must be at for the END of the travel segment
            # Note, camera calculations need to be done in real position to correctly get the angle
            # Reminder that real positions are the absolute system position ignoring any home coord systems that were set. 
            previous_real_pos = (real_positions["X"], real_positions["Y"])
            if "X" not in axes_displacements:
                axes_displacements.update({"X":0.0})
            if "Y" not in axes_displacements:
                axes_displacements.update({"Y":0.0})
            target_pos=(real_positions["X"] + axes_displacements["X"], real_positions["Y"] + axes_displacements["Y"])
            unmapped_end_cam_ang = calc_camera_pos(mode="2pos", previous_pos=previous_real_pos, target_pos=target_pos, units="deg")
            end_cam_ang = convert_ang_range(x=unmapped_end_cam_ang, units="deg")
            end_cam_displacement = (end_cam_ang - real_positions[CAM_AXIS_NAME])    # finds displacement without crossing -180/180 line
            axes_displacements.update({CAM_AXIS_NAME:end_cam_displacement})

            # Calculates time to execute WHOLE command 
            xyz_dist = np.sqrt(xyz_dist_sq)
            print("xyz_dist = {}".format(xyz_dist))
            try:
                tot_time = xyz_dist / speed
                print("tot_time = {}".format(tot_time))
                print("speed = {}".format(speed))
            except:
                if speed == 0:
                    raise RuntimeError("speed is 0")
                else:
                    raise RuntimeError("speed is None")

            # In case where camera does not have enough time to reach the target goal, the travel rate
            # will slow in order to let it reach the target in time. 
            cam_time_limited = False
            if (tot_time * MAX_CAM_SPEED) < np.abs(end_cam_displacement):
                cam_time_limited = True
                original_target_speed = speed
                speed = np.floor(100*np.abs(end_cam_displacement) / MAX_CAM_SPEED)/100.0
                tot_time = xyz_dist / speed

            # Calculates new speed for including camera if camera axis is dominant
            if camera_dominant:
                # A little bit inefficient, but robust to use the function (calculates the new required speed for only case in which no need to discretize)
                if mode.lower() == "relative":
                    new_speed = calc_new_speed(cmd_info=cmd_info, speed=speed, positions=positions, mode=mode, target_cam_ang=end_cam_displacement)
                elif mode.lower() == "absolute":
                    new_speed = calc_new_speed(cmd_info=cmd_info, speed=speed, positions=positions, mode=mode, target_cam_ang=end_cam_ang)
                else:
                    raise RuntimeError("Using incorrect `mode`; `mode` = {}".format(mode))
            else:
                new_speed = speed
            
            # No need to discretize if whole command is already within threshold
            num_disc_parts = int(np.ceil(tot_time / disc_t)) 
            if num_disc_parts <= 1:     
                new_ln = prefix + " "
                for axis_name in cmd_info:
                    if axis_name == "F":
                        new_ln = new_ln + "F" + str(np.round(new_speed*SPEED_FACTOR, ROUND_DECIMAL)) + " "
                    else:
                        new_ln = new_ln + axis_name + str(np.round(cmd_info[axis_name], ROUND_DECIMAL))
                if CAM_AXIS_NAME not in cmd_info:
                    if mode.lower() == "absolute":
                        new_ln = new_ln + CAM_AXIS_NAME + str(axes_displacements[CAM_AXIS_NAME] + positions[CAM_AXIS_NAME]) + " "
                    elif mode.lower() == "relative":
                        new_ln = new_ln + CAM_AXIS_NAME + str(axes_displacements[CAM_AXIS_NAME]) + " "
                    else:
                        raise RuntimeError("wrong mode")
                if comment:
                    new_ln = new_ln + comment
                last_cmd_info = cmd_info
                if debug:
                    print("Writing--Camera does not need to discretize to adjust position")
                mv_cnt += 1
                w_file_ob.write(new_ln)
                if cam_time_limited:
                    speed = original_target_speed
                continue

            ## Entering Camera adjustment period:
            cam_aligned = False                                     # Is the camera in the right position relative to where it is supposed to be?
            cam_disc_t = min(disc_t, CAMERA_DISC_T)                 # Find the smallest to discretize the camera commands by
            expired_time = 0.0                                      # expired time since start of this move (real time, not execution time)
            cam_disc_parts = int(np.ceil(tot_time / cam_disc_t))    # number of sections that the camera discretization is broken into
            act_cam_t = tot_time / float(cam_disc_parts)            # actual amount of time that will expire for each camera discretization
            cam_alignment_it = 0                                    # number of iterations required for camera alignment

            # If it takes more than 2 secs at target speed to reach the correct position, go at faster speed
            cam_speed = MAX_CAM_SPEED if cam_time_limited or (np.abs(axes_displacements[CAM_AXIS_NAME]/2.0) > TARGET_CAM_SPEED) else TARGET_CAM_SPEED

            while not(cam_aligned) and (tot_time - expired_time) > FLOAT_RND_ERR_THRESHOLD:
                frac_time_expire = expired_time/tot_time        # fraction [0, 1] of amount of time given for this command has already expired
                cam_disc_frac = 1.0/float(cam_disc_parts)       # fraction [0, 1] of amount of time of 1 discretization line of initial command
                cam_alignment_it += 1

                next_x = real_positions["X"] + (cam_disc_frac*axes_displacements["X"])
                next_y = real_positions["Y"] + (cam_disc_frac*axes_displacements["Y"])

                # Calculations for next camera angle and the displacement to get there. Very similar to above
                target_cam_ang = convert_ang_range(x=calc_camera_pos(previous_pos=(real_positions["X"], real_positions["Y"]), target_pos=(next_x, next_y), mode="2pos", units="deg"), units="deg")
                back_cross = True if np.abs(target_cam_ang - real_positions[CAM_AXIS_NAME]) > 180.0 else False # Detects crossing -180 <-> 180 
                target_cam_displacement = (target_cam_ang - real_positions[CAM_AXIS_NAME])  # calculates displacement without crossing -180/180
                target_cam_ang = real_positions[CAM_AXIS_NAME] + target_cam_displacement
                cam_displacement_sign = 1 if target_cam_displacement > 0 else -1
                
                # Calculates the actual next angle for the camera
                can_reach_ang_in_time = True if np.abs(target_cam_displacement/act_cam_t) <= cam_speed else False
                cam_aligned = True if can_reach_ang_in_time else False
                next_cam_ang_disp = target_cam_displacement if can_reach_ang_in_time else (cam_displacement_sign*cam_speed*act_cam_t)
                
                ### Creates discretization line START ###
                cmd_info_eqv = {}       # equivalent form to cmd_info for calc-ing new required speed
                result_ln = prefix + " "          # resulting string that will be printed
                for axis_name in positions:
                    if axis_name in axes_displacements:
                        if axis_name != CAM_AXIS_NAME:
                            disc_displacement = np.round(axes_displacements[axis_name]*(cam_disc_frac), 3)
                        else:
                            disc_displacement = np.round(next_cam_ang_disp, 3)
                        if mode.lower() == "absolute":
                            cmd_info_eqv.update({axis_name:disc_displacement+positions[axis_name]})
                            result_ln = result_ln + axis_name + str(np.round(disc_displacement + positions[axis_name],3)) + " "
                        elif mode.lower() == "relative":
                            cmd_info_eqv.update({axis_name:disc_displacement})
                            result_ln = result_ln + axis_name + str(np.round(disc_displacement,3)) + " "
                        else:
                            raise RuntimeError("mode set incorrectly to \"{}\"".format(mode))
                    else:
                        if mode.lower() == "absolute":
                            result_ln = result_ln + axis_name + str(np.round(positions[axis_name],3)) + " "
                        elif mode.lower() == "relative":
                            result_ln = result_ln + axis_name + str(0.0) + " "
                        else:
                            raise RuntimeError("mode set incorrectly to \"{}\"".format(mode))

                # Calculates new speed for including camera if camera axis is dominant
                if camera_dominant:
                    if mode.lower() == "relative":
                        new_speed = calc_new_speed(cmd_info=cmd_info_eqv, speed=speed, positions=positions, mode=mode, target_cam_ang=next_cam_ang_disp)
                    elif mode.lower() == "absolute":
                        if debug:
                            print("positions = {}\nmode = {}".format(positions, mode))
                            print("cmd_info_eqv = {}\ntarget_cam_ang = {}\nspeed = {}".format(cmd_info_eqv, (next_cam_ang_disp + positions[CAM_AXIS_NAME]), speed))
                        new_speed = calc_new_speed(cmd_info=cmd_info_eqv, speed=speed, positions=positions, mode=mode, target_cam_ang=next_cam_ang_disp + positions[CAM_AXIS_NAME])
                        print("new_speed = {}".format(new_speed))
                    else:
                        raise RuntimeError("Using incorrect `mode`; `mode` = {}".format(mode))
                else:
                    new_speed = speed

                result_ln = result_ln + "F" + str(np.round(new_speed*SPEED_FACTOR,3)) + " "
                if expired_time == 0.0 and comment:
                    result_ln = result_ln + comment
                if expired_time == 0.0 and debug: 
                    result_ln = result_ln + "\n"
                    if comment:
                        subline = line.replace(comment, "")
                    else:
                        subline = line[:-1]
                    w_file_ob.write("; Discretization for: \"{}\" START \\/ [\n".format(subline))
                    w_file_ob.write("; Camera Alignment Start\n".format(subline))
                elif ((expired_time + act_cam_t)/tot_time) >= 1.0 and debug:
                    w_file_ob.write(result_ln + "\n")
                    if comment:
                        subline = line.replace(comment, "")
                    else:
                        subline = line[:-1]
                    result_ln = "; Discretization for: \"{}\" END   /\\ ]\n".format(subline)
                else: 
                    result_ln = result_ln + "\n"
                expired_time += act_cam_t
                w_file_ob.write(result_ln)
                if can_reach_ang_in_time:
                    w_file_ob.write("; Camera Alignment End\n".format(subline))
                ### Creates discretization line END ###

                # Sets values for next iteration by updating all positions and expired time
                for axis_name in positions:
                    if axis_name in axes_displacements:
                        if mode.lower() == "absolute":
                            if axis_name == CAM_AXIS_NAME:
                                real_positions[CAM_AXIS_NAME] = real_positions[CAM_AXIS_NAME] + next_cam_ang_disp
                                positions[CAM_AXIS_NAME] = positions[CAM_AXIS_NAME] + next_cam_ang_disp
                            else:
                                real_positions[axis_name] = real_positions[axis_name] + (cam_disc_frac * axes_displacements[axis_name])
                                positions[axis_name] = positions[axis_name] + (cam_disc_frac * axes_displacements[axis_name])
                        elif mode.lower() == "relative":
                            if axis_name == CAM_AXIS_NAME:
                                real_positions[CAM_AXIS_NAME] = next_cam_ang_disp
                                positions[CAM_AXIS_NAME] = next_cam_ang_disp
                            else:
                                real_positions[axis_name] = (cam_disc_frac * axes_displacements[axis_name])
                                positions[axis_name] = (cam_disc_frac * axes_displacements[axis_name])
                        else:
                            raise RuntimeError("incorrectly set `mode` parameter")

                ### REITERATE WHILE LOOP FOR NEXT STEP OF CAMERA DISCRETIZATION

            print("cam_time_limited={}".format(cam_time_limited))
            if cam_time_limited:
                speed = original_target_speed
                print("speed = {}".format(speed))

            remaining_time = tot_time - expired_time

            # Checks if all time was used up in getting camera aligned
            if (remaining_time) <= FLOAT_RND_ERR_THRESHOLD: 
                last_cmd_info = cmd_info
                continue    

            disc_parts = int(np.ceil((remaining_time) / disc_t)) 
            act_disc_t = remaining_time / float(disc_parts) # time in seconds for each discretized command
            frac_of_tot_t = act_disc_t / tot_time           # fraction [0,1] of total time expired

            ### START NOT DISCRETIZING AFTER CAMERA ALIGNED ###
            result_ln = prefix + " "
            cmd_info_eqv = {}
            if disc_parts <= 1:     # No need to discretize if it is already within threshold
                for axis_name in positions:
                    if axis_name in cmd_info:
                        if axis_name == CAM_AXIS_NAME:
                            back_cross = True if np.abs(end_cam_ang - real_positions[CAM_AXIS_NAME]) > 180.0 else False # Detects crossing -180 <-> 180 
                            target_cam_displacement = (end_cam_ang - positions[CAM_AXIS_NAME])
                            # calculates the shortest displacement path
                            disc_displacement = np.round(target_cam_displacement, 3)
                        else:
                            disc_displacement = np.round(cmd_info[axis_name] - positions[axis_name], 3)
                        if mode.lower() == "absolute":
                            cmd_info_eqv.update({axis_name:disc_displacement+positions[axis_name]})
                            result_ln = result_ln + axis_name + str(np.round(disc_displacement + positions[axis_name],3)) + " "
                        elif mode.lower() == "relative":
                            cmd_info_eqv.update({axis_name:disc_displacement})
                            result_ln = result_ln + axis_name + str(np.round(disc_displacement,3)) + " "
                        else:
                            raise RuntimeError("mode set incorrectly to \"{}\"".format(mode))

                # Calculates new speed for including camera if camera axis is dominant
                if camera_dominant:       
                    if mode.lower() == "relative":
                        new_speed = calc_new_speed(cmd_info=cmd_info_eqv, speed=speed, positions=positions, mode=mode, target_cam_ang=target_cam_displacement)
                    elif mode.lower() == "absolute":
                        new_speed = calc_new_speed(cmd_info=cmd_info_eqv, speed=speed, positions=positions, mode=mode, target_cam_ang=target_cam_displacement + positions[CAM_AXIS_NAME])
                    else:
                        raise RuntimeError("Using incorrect `mode`; `mode` = {}".format(mode))
                else:
                    new_speed = speed
                
                result_ln = result_ln + "F" + str(np.round(new_speed*SPEED_FACTOR,3)) + " "
                if comment:
                    result_ln = result_ln + comment
                result_ln = result_ln + "\n"
                w_file_ob.write(result_ln)
                update_pos(cmd_info=cmd_info_eqv, positions=positions, real_positions=real_positions, mode=mode)
                continue
            ### END NOT DISCRETIZING AFTER CAMERA ALIGNED ###

            ### START DISCRETIZING AFTER CAMERA ALIGNED ###
            for disc_idx in range(disc_parts):
                result_ln = prefix + " "
                frac_of_tot_t = (remaining_time/tot_time)*(1.0/disc_parts)
                next_x = real_positions["X"] + (frac_of_tot_t*axes_displacements["X"])
                next_y = real_positions["Y"] + (frac_of_tot_t*axes_displacements["Y"])
                cmd_info_eqv = {}
                for axis_name in positions:

                    # Assume that the discretized command should be small enough such that the camera can always get to the next position
                    # When moving at the slower camera speed 
                    target_cam_ang = convert_ang_range(x=calc_camera_pos(previous_pos=(real_positions["X"], real_positions["Y"]), target_pos=(next_x, next_y), mode="2pos", units="deg"), units="deg")
                    back_cross = True if np.abs(target_cam_ang - real_positions[CAM_AXIS_NAME]) > 180.0 else False # Detects crossing -180 <-> 180 
                    target_cam_displacement = (-1.0 if real_positions[CAM_AXIS_NAME] < 0.0 else 1.0) * ((360.0 if back_cross else 0.0) + \
                                               (-1.0 if real_positions[CAM_AXIS_NAME] < 0.0 else 1.0)*(target_cam_ang - real_positions[CAM_AXIS_NAME]))

                    if axis_name in axes_displacements:
                        if axis_name == CAM_AXIS_NAME:
                            disc_displacement = np.round(target_cam_displacement, 3)
                        else:
                            disc_displacement = np.round(axes_displacements[axis_name]*frac_of_tot_t, 3)
                        if mode.lower() == "absolute":
                            cmd_info_eqv.update({axis_name:disc_displacement+positions[axis_name]})
                            result_ln = result_ln + axis_name + str(np.round(disc_displacement + positions[axis_name],3)) + " "
                        elif mode.lower() == "relative":
                            cmd_info_eqv.update({axis_name:disc_displacement})
                            result_ln = result_ln + axis_name + str(np.round(disc_displacement,3)) + " "
                        else:
                            raise RuntimeError("mode set incorrectly to \"{}\"".format(mode))
                    else:
                        if mode.lower() == "absolute":
                            result_ln = result_ln + axis_name + str(np.round(positions[axis_name],3)) + " "
                        elif mode.lower() == "relative":
                            result_ln = result_ln + axis_name + str(0.0) + " "
                        else:
                            raise RuntimeError("mode set incorrectly to \"{}\"".format(mode))

                # Calculates new speed for including camera if camera axis is dominant
                if camera_dominant:
                    if mode.lower() == "relative":
                        new_speed = calc_new_speed(cmd_info=cmd_info_eqv, speed=speed, positions=positions, mode=mode, target_cam_ang=target_cam_displacement)
                    elif mode.lower() == "absolute":
                        new_speed = calc_new_speed(cmd_info=cmd_info_eqv, speed=speed, positions=positions, mode=mode, target_cam_ang=target_cam_displacement + positions[CAM_AXIS_NAME])
                    else:
                        raise RuntimeError("Using incorrect `mode`; `mode` = {}".format(mode))
                else:
                    new_speed = speed

                result_ln = result_ln + "F" + str(np.round(new_speed*SPEED_FACTOR,3)) + " "
                if disc_idx == 0 and comment and cam_alignment_it == 1:
                    result_ln = result_ln + comment
                if disc_idx == 0 and debug and cam_alignment_it == 1: 
                    result_ln = result_ln + "\n"
                    if comment:
                        subline = line.replace(comment, "")
                    else:
                        subline = line[:-1]
                    w_file_ob.write("; Discretization for: \"{}\" START \\/ [\n".format(subline))
                elif disc_idx == disc_parts - 1 and debug:
                    w_file_ob.write(result_ln + "\n")
                    if comment:
                        subline = line.replace(comment, "")
                    else:
                        subline = line[:-1]
                    result_ln = "; Discretization for: \"{}\" END   /\\ ]\n".format(subline)
                else: 
                    result_ln = result_ln + "\n"
                w_file_ob.write(result_ln)
                update_pos(cmd_info=cmd_info_eqv, positions=positions, real_positions=real_positions, mode=mode)
            
            ### END DISCRETIZING AFTER CAMERA ALIGNED ###
                
            write_ln = None
            last_cmd_info = cmd_info
            for axis_name in positions:
                if axis_name not in last_cmd_info:
                    last_cmd_info.update({axis_name:positions[axis_name]})
            mv_cnt = mv_cnt + 1
        else:
            if debug == True:
                pass
            write_ln = line

        if write_ln:    
            w_file_ob.write(write_ln)

    w_file_ob.close()

##############################################################
#                     HELPER FUNCTIONS                       #
##############################################################


def get_prefix(line: str):
    """
    Description:
        Returns prefix (letter plus number corresponding to a specific G-code 
        command) for a line of G-code. Returns None for blank lines / comments

    Parameters:
        line        : str
            String containing the line of g-code

    Returns:
        String for just the prefix
    """

    if not (line[0] == "M" or line[0] == "G"):
        return None 
    
    prefix_end = 1
    found_end = False
    while not found_end:
        if not (line[prefix_end] == ";" or line[prefix_end] == " "):
            prefix_end = prefix_end + 1
        else:
            found_end = True
    
    return line[0:prefix_end]


def update_pos(cmd_info, positions, real_positions, mode):
    """
    Updates positions of all axes according to the mode
    
    Parameters:
    -----------
        cmd_info    : Dict
            result from an extract_XXXinfo() method
        positions   : Dict
            Contains all axis names as keys and values are their current positions in absolute coords
        mode        : str ("Absolute" or "Relative" only)
            Tells system how to interpret the positions
    """
    if mode.lower() == "relative":
        for axis_name in AXES_LIST:
            if axis_name in cmd_info:
                real_positions[axis_name] = real_positions[axis_name] + cmd_info[axis_name]
                positions[axis_name] = positions[axis_name] + cmd_info[axis_name]
            else:
                real_positions[axis_name] = real_positions[axis_name]   
                positions[axis_name] = positions[axis_name]             
    elif mode.lower() == "absolute":
        for axis_name in AXES_LIST:
            if axis_name in cmd_info:
                displacement = cmd_info[axis_name] - positions[axis_name]
                real_positions[axis_name] = displacement + real_positions[axis_name]
                positions[axis_name] = cmd_info[axis_name]
            else:
                real_positions[axis_name] = real_positions[axis_name]   
                positions[axis_name] = positions[axis_name]             
    else:
        raise RuntimeError("Incorrect mode provided to update_pos()")

    return

def extract_G1info(line: str):
    """
    Extracts the important info from a G1 or G0 command.
    Assumes formatting of: 
        "G1 X10 Y50 Z15 U13.33 F15.5; Comment here"

    Parameters:
        line        : str
            String containing the line of g-code
        mode        : "relative" or "absolute"
            Tells system how to interpret commands

    Returns:
        A Dict containing {"Axis name":float}
    """

    result = {}
    comment = None
    word_idx_start = 3 # "G1_"
    word_idx_end = word_idx_start
    last_char_ws = True     # True if the last character found was whitespace
    found_end=False         # True if the end of the line or a comment is reached
    comment_found = False   # True if a comment is found
    while not found_end:
        if (line[word_idx_end] == ";"):
            found_end = True
            comment_found = True
            if (" " not in line[word_idx_start:word_idx_end] or \
                  "\t" not in line[word_idx_start:word_idx_end]) and \
                    len(line[word_idx_start:word_idx_end]) > 0:
                result.update({line[word_idx_start]:float(line[word_idx_start+1:word_idx_end])})
        elif (line[word_idx_end] == "\n"):
            found_end = True
            if (" " not in line[word_idx_start:word_idx_end] or \
                  "\t" not in line[word_idx_start:word_idx_end]) and \
                    len(line[word_idx_start:word_idx_end]) > 0:
                result.update({line[word_idx_start]:float(line[word_idx_start+1:word_idx_end])})
        elif (line[word_idx_end] == " " or line[word_idx_end] == "\t"):
            if not last_char_ws:
                result.update({line[word_idx_start]:float(line[word_idx_start+1:word_idx_end])})
            word_idx_start = word_idx_end + 1
            word_idx_end = word_idx_start
            last_char_ws = True
        else:
            word_idx_end = word_idx_end + 1
            last_char_ws = False

    if comment_found:
        comment = line[word_idx_end:]
    
    return result, comment


def extract_G92info(line: str):
    """
    Extracts the important info from a G92 command.
    G92 is used for setting the position of an axis WITHOUT moving
    Assumes formatting of: 
        "G92 X10 Y50 Z15 A13.33 ; Comment here"

    Parameters:
        line        : str
            String containing the line of g-code

    Returns:
        [0]: A Dict containing {"Axis name":float}
        [1]: A str representing supplied comment
    """

    result = {}
    comment = None
    word_idx_start = 4 # "G92_"
    word_idx_end = word_idx_start
    last_char_ws = True     # True if the last character found was whitespace
    found_end=False         # True if the end of the line or a comment is reached
    comment_found = False   # True if a comment is found
    while not found_end:
        if (line[word_idx_end] == ";"):
            found_end = True
            comment_found = True
            if (" " not in line[word_idx_start:word_idx_end] or \
                  "\t" not in line[word_idx_start:word_idx_end]) and \
                    len(line[word_idx_start:word_idx_end]) > 0:
                result.update({line[word_idx_start]:float(line[word_idx_start+1:word_idx_end])})
        elif (line[word_idx_end] == "\n"):
            found_end = True
            if (" " not in line[word_idx_start:word_idx_end] or \
                  "\t" not in line[word_idx_start:word_idx_end]) and \
                    len(line[word_idx_start:word_idx_end]) > 0:
                result.update({line[word_idx_start]:float(line[word_idx_start+1:word_idx_end])})
        elif (line[word_idx_end] == " " or line[word_idx_end] == "\t"):
            if not last_char_ws:
                result.update({line[word_idx_start]:float(line[word_idx_start+1:word_idx_end])})
            word_idx_start = word_idx_end + 1
            word_idx_end = word_idx_start
            last_char_ws = True
        else:
            word_idx_end = word_idx_end + 1
            last_char_ws = False

    if comment_found:
        comment = line[word_idx_end:]
    
    print("extractG92_result = {}".format(result))
    return result, comment


def calc_camera_pos(mode: str=None, previous_pos: Tuple=None, target_pos: Tuple=None, dir: Tuple=None, units: str=None):
    """
    calculates the desired camera position from 2 possible sets of input:
    if `mode` = "2pos": provide `target_pos` and `previous_pos`
    if `mode` = "posdir": provide `target_pos` and `dir` for the direction the SCARA will be moving when arriving at the target position
    
    if `units` = "rad" -> returns result in radians
    if `units` = "deg" -> returns result in degrees

    Requires target_pos and previous_pos to be in absolute coordinates not relative coords
    """

    if mode == "2pos":
        if not (target_pos and previous_pos and dir==None):
            raise RuntimeError("`mode`=\"2pos\" must have both `target_pos` and `previous_pos` defined while `dir`=None")
        dir = (target_pos[0]-previous_pos[0], -1.0*(target_pos[1]-previous_pos[1]))
        target_pos = (target_pos[0], -1.0*target_pos[1])
        dir = dir/np.linalg.norm(np.array(dir))
    elif mode == "posdir":
        if not (target_pos and dir and previous_pos==None):
            raise RuntimeError("`mode`=\"posdir\" must have both `target_pos` and `dir` defined while `previous_pos`=None")
        dir = dir/np.linalg.norm(np.array(dir))
        dir = (dir[0], -1.0*dir[1])
        target_pos = (target_pos[0], -1.0*target_pos[1])
    else:
        raise RuntimeError("incorrect `mode` provided. Must be \"2pos\" or \"posdir\"")

    # Camera angle requirement due to the direction of the motion
    cam_ang_vel = np.arctan2(-1*dir[1], -1*dir[0]) 

    # target position values
    x0 = float(target_pos[0])
    y0 = float(target_pos[1])

    ### Calculations from SCARA DEMO on DESMOS START ###
    scara_arm_l1 = SCARA_ARM_L1     # (mm) 
    scara_arm_l2 = SCARA_ARM_L2     # (mm)
    theta1 = np.arctan2(y0, x0) # y0 / x0

    theta2 = np.arccos(((scara_arm_l1**2) + (scara_arm_l2**2) - (x0**2) - (y0**2)) / (2*scara_arm_l1*scara_arm_l2))
        
    r = np.sqrt((scara_arm_l1**2) + (scara_arm_l2**2) - 2*scara_arm_l1*scara_arm_l2*np.cos(theta2))
    theta_l_2 = np.arccos(((r**2)+(scara_arm_l1**2)-(scara_arm_l2**2)) / (2*scara_arm_l1*r))
    
    alpha = theta1 - theta_l_2
    beta = np.pi - theta2
    # Camera angle requirement due to the position required for the camera
    cam_ang_pos = alpha + beta
    ### Calculations from SCARA DEMO on DESMOS END   ###

    abs_target_cam_ang = cam_ang_vel - cam_ang_pos 
    if units == None:
        raise RuntimeError("`units` cannot be None")
    elif units.lower() == "rad":
        pass # by default units are in radians
    elif units.lower() == "deg":
        abs_target_cam_ang = abs_target_cam_ang * RAD_2_DEG
    else:
        raise RuntimeError("units are not correctly provided")

    return abs_target_cam_ang


def convert_ang_range(x:float, units:str = "rad"):
    """
    maps any angular position to a (-180 deg <-> 180) deg range
    gives result in the same units as input
    """
    if units.lower() == "rad":
        if float(x) == np.pi:
            result = np.pi
        else:
            result = ((x-np.pi)%(2*np.pi))-np.pi
    elif units.lower() == "deg":
        if float(x) == 180.0:
            result = 180.0
        else:
            result = ((x-180.0)%360.0)-180.0
    else:
        raise RuntimeError("units are not correctly provided. They are given as \"{}\"".format(units))
    return result

def extend_ang_range(x:float, units:str = "rad"):
    """
    extends the values of the [-180, 180] range to [-360, 360] deg range
    takes input from the [-180, 180] range and extends on opposite side of the range

    Ex: x=90,   units="deg" -> return -270
    Ex: x=-90,  units="deg" -> return 270
    Ex: x=270,  units="deg" -> return 270
    Ex: x=-270, units="deg" -> return -270
    """
    if units.lower() == "rad":
        if float(x) > 2.0*np.pi or float(x) < -2.0*np.pi:
            raise RuntimeError("Provided X is out of [-360, 360] range: x={}".format(x))
        if float(x) >= np.pi or float(x) <= -np.pi:
            result = float(x)
        else:
            result = -1.0*float(x)
    elif units.lower() == "deg":
        if float(x) > 360.0 or float(x) < -360.0:
            raise RuntimeError("Provided X is out of [-360, 360] range: x={}".format(x))
        if float(x) >= 180.0 or float(x) <= -180.0:
            result = float(x)
        else:
            result = -1.0*float(x)
    else:
        raise RuntimeError("units are not correctly provided. They are given as \"{}\"".format(units))
    return result


def calc_new_speed(cmd_info, speed, positions, mode, target_cam_ang) -> float:
    """
    Given the target speed for the provided axes, calculates the new (increased) speed
    required in order to include a "Dominant" camera axis while maintaining the desired
    speed for the other axes. 
    """
    xyz_speed = speed
    xyz_dist = 0.0
    tot_dist = 0.0
    for axis_name in AXES_LIST:
        if axis_name == "Z" or axis_name == "Y" or axis_name == "X":
            if axis_name in cmd_info:
                if mode.lower() == "relative":
                    xyz_dist = np.sqrt((xyz_dist**2)+(cmd_info[axis_name]**2))
                    tot_dist = np.sqrt((tot_dist**2)+(cmd_info[axis_name]**2)) 
                elif mode.lower() == "absolute":
                    xyz_dist = np.sqrt((xyz_dist**2)+((cmd_info[axis_name]-positions[axis_name])**2))
                    tot_dist = np.sqrt((tot_dist**2)+((cmd_info[axis_name]-positions[axis_name])**2))
                else:
                    raise RuntimeError("Using incorrect mode")
        elif axis_name == CAM_AXIS_NAME:
            if axis_name in cmd_info:
                if mode.lower() == "relative":
                    tot_dist = np.sqrt((tot_dist**2)+(cmd_info[axis_name]**2))
                elif mode.lower() == "absolute":
                    tot_dist = np.sqrt((tot_dist**2)+((cmd_info[axis_name]-positions[axis_name])**2))
                else:
                    raise RuntimeError("Using incorrect mode")
            else:
                if mode.lower() == "relative":
                    tot_dist = np.sqrt((tot_dist**2)+(target_cam_ang**2))
                elif mode.lower() == "absolute":
                    tot_dist = np.sqrt((tot_dist**2)+((target_cam_ang-positions[axis_name])**2))
                else:
                    raise RuntimeError("Using incorrect mode")
        elif axis_name == "E":
            if E_AXIS_DOMINANT:
                if mode.lower() == "relative":
                    xyz_dist = np.sqrt((xyz_dist**2)+(cmd_info[axis_name]**2))
                    tot_dist = np.sqrt((tot_dist**2)+(cmd_info[axis_name]**2))
                elif mode.lower() == "absolute":
                    xyz_dist = np.sqrt((xyz_dist**2)+((cmd_info[axis_name]-positions[axis_name])**2))
                    tot_dist = np.sqrt((tot_dist**2)+((cmd_info[axis_name]-positions[axis_name])**2))
                else:
                    raise RuntimeError("Using incorrect mode")

        else:
            raise RuntimeError("Logic Error in calc_new_speed(). Not found axis name")
        
    new_speed = (tot_dist/xyz_dist) * speed
    return new_speed


if __name__ == '__main__': 
    main()