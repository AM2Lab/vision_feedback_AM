# Vision-based Feedback Control for Robotic Additive Manufacturing 
A vision-based control system is developed to adjust the width and height of the extruded filaments in robotic AM with two independent PID controllers. The code also includes camera adjustments by the vision alignment mechanism.<br>
__camera.py__ interacts with the Intel RealSense D405 Camera.<br>
__feedback.py__ performs real-time feedback control.<br>
__gcode_mods.py__ modifies the G-code commands as an adjustment output of the feedback control system. The input file is a standard SCARA G-code, and the output is a discretized G-code. <br>
__image_module.py__ is the image-processing module. 

