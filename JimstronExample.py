#!/usr/bin/env python3.8
"""Jimstron test example. Do not delete. New files can be create via copying this
 either manual or creating a new test in the GUI"""

import random, yaml, json, datetime
import rospy

from std_msgs.msg import Bool
from std_msgs.msg import String
from std_msgs.msg import Float32

from std_srvs.srv import SetBool
from std_srvs.srv import Trigger
from std_srvs.srv import TriggerResponse

from jimstron.srv import SetFloat
from jimstron.srv import SetString

from click_plc_ros.srv import SetRegister
from click_plc_ros.srv import GetRegister

from laumas_ros.srv import SetFloat as SetFloatLaumas

# ROS Image message -> OpenCV2 image converter, OpenCV2 for saving an image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2

#Globals
test_stop_flag = False

position = None
velocity = None
torque = None

servo_status = None
servo_motion_complete = False

force = None
force_raw = None
gpio_data = None

image_msg = None

##################################################################################
#                                    Callbacks
##################################################################################

def position_callback(msg):
    global position
    position = float(msg.data)

def velocity_callback(msg):
    global velocity
    velocity = float(msg.data)

def torque_callback(msg):
    global torque
    torque = float(msg.data)

def servo_status_callback(msg):
    global servo_status_callback
    servo_status_callback = json.loads(msg.data)

def servo_moving_callback(msg):
    global servo_motion_complete
    servo_motion_complete = bool(msg.data)

def gpio_data_update(msg):
    global gpio_data
    gpio_data = json.loads(msg.data)

def load_cell_raw_update(msg):
    global force_raw 
    force_raw = float(msg.data)

def load_cell_force_update(msg):
    global force, max_pulloff_force
    force = float(msg.data)

def image_callback(msg:CompressedImage):
    global image_msg
    image_msg = msg

def test_stop_callback(msg):
    global test_stop_flag
    test_stop_flag = True
    test_status_pub_.publish("STOPPING")
    res = TriggerResponse()
    res.success = True
    return res

##################################################################################
#                       Publishers, Subscribers, Services
#               * See rostopic list for more individual topics 
##################################################################################

### ROS handlers for the servo motor
rospy.init_node("jimstron_test", log_level=rospy.INFO, anonymous=False, disable_signals=True)

#Test
test_status_pub_ =  rospy.Publisher("test/status", String, queue_size=5, latch=True)
rospy.Service("test/stop", Trigger, test_stop_callback)

#Logger
logging_start_ = rospy.ServiceProxy("logger/start", Trigger)
logging_stop_ = rospy.ServiceProxy("logger/stop", Trigger)
logging_save_ = rospy.ServiceProxy("logger/save", SetString)

# Servo - Subscribers
rospy.Subscriber("jimstron/position", Float32, position_callback)
rospy.Subscriber("jimstron/velocity", Float32, velocity_callback)
rospy.Subscriber("jimstron/torque", Float32, torque_callback)
rospy.Subscriber("servo/status/all", String, servo_status_callback)
rospy.Subscriber("servo/status/move_cmd_complete", Bool, servo_moving_callback) 
# Servo - Services
servo_set_acc_lim_ = rospy.ServiceProxy("servo/set_acc_lim", SetFloat)
servo_set_vel_lim_ = rospy.ServiceProxy("servo/set_vel_lim", SetFloat)
servo_set_decel_lim_ = rospy.ServiceProxy("servo/set_decel_lim", SetFloat)
servo_absolute_move_ = rospy.ServiceProxy("servo/absolute_move", SetFloat)
servo_relative_move_ = rospy.ServiceProxy("servo/relative_move", SetFloat)
servo_velocity_move_ = rospy.ServiceProxy("servo/velocity_move", SetFloat)
servo_reset_ = rospy.ServiceProxy("servo/reset", Trigger)
servo_enable_ = rospy.ServiceProxy("servo/enable", SetBool)
servo_stop_ = rospy.ServiceProxy("servo/stop", Trigger)
servo_cancel_ = rospy.ServiceProxy("servo/cancel", Trigger)
servo_home_ = rospy.ServiceProxy("servo/home", Trigger)

#Load cell
rospy.Subscriber("load_cell/raw", Float32, load_cell_raw_update)
rospy.Subscriber("load_cell/force", Float32, load_cell_force_update)
load_cell_zero_ = rospy.ServiceProxy("load_cell/zero", Trigger)
load_cell_calibrate_ = rospy.ServiceProxy("load_cell/calibrate", SetFloatLaumas)

#GPIO
rospy.Subscriber("plc/all_data", String, gpio_data_update)
plc_set_ = rospy.ServiceProxy("plc/set", SetRegister)
plc_get_ = rospy.ServiceProxy("plc/get", GetRegister)

#Camera
camera_1_save_image = rospy.ServiceProxy("camera_1/save_image", SetString)
camera_1_start_timelapse = rospy.ServiceProxy("camera_1/timelapse/start", SetString)
camera_1_stop_timelapse = rospy.ServiceProxy("camera_1/timelapse/stop", Trigger)
# camera_2_save_image = rospy.ServiceProxy("camera_2/save_image", SetString)
# camera_2_start_timelapse = rospy.ServiceProxy("camera_2/timelapse/start", SetString)
# camera_2_stop_timelapse = rospy.ServiceProxy("camera_2/timelapse/stop", Trigger)

##################################################################################
#                                Helper functions 
##################################################################################

def ros_msg_to_json(msg):
    y = yaml.load(str(msg), yaml.SafeLoader)
    return json.loads(json.dumps(y))

# def save_image(dir, name=None, extension=".jpeg"):
#     global image_msg
#     if not os.path.isdir(dir):
#         os.makedirs(dir)
        
#     if image_msg:
#         filename = name if name else str(image_msg.header.stamp)
#         subscribed_image = CvBridge().compressed_imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
#         cv2.imwrite(os.path.join(dir, filename+extension), subscribed_image)
#         image_msg = None  #Clear image temp to stop the same img getting saved next call
#     else:
#         rospy.logwarn("No image ready to save")

def wait(break_func=None, *args):
    rospy.sleep(0.1)    # Avoid race condition, really small moves might be effected by this TODO: remove need for this
    rate = rospy.Rate(50)
    while not servo_motion_complete:
        # Check if break function exists. Stop servo and return on True
        if break_func and break_func(*args): 
            servo_stop_()
            return False # Wait exited early
        rate.sleep()
    return True # Wait completed succesfully

def abs_limit_force(force_limit):
    if abs(force) >= force_limit:
        rospy.logwarn("Force limit exceeded!") 
        return True
    return False

##################################################################################
#                                ADD TEST CODE HERE 
##################################################################################

def pre_test():
    """
    Runs outside of logging. Used to setup positon, etc before looging starts
    """
    pass

def test():
    """
    Main loop. Everything is logged during this loop
    """
    servo_set_vel_lim_(10)		# Set servo velocity to 10mm/s
    servo_absolute_move_(-100) 	# Move down 100mm from home position
    if not wait(abs_limit_force, 10):	# Exit test early if force is exceeded
    	return
    	
    while not rospy.is_shutdown() and not test_stop_flag:
        servo_relative_move_(-50) 	# Move down 50mm
        wait(abs_limit_force, 30)  	# Wait for motor to finish movement, while checking force
        servo_absolute_move_(-100) 	# Move back to start
        wait()

        if (gpio_data):
            plc_set_("C1", [not gpio_data["C1"]]) #Toggle GPIO C1 pin (cabin light)
            rospy.loginfo(gpio_data)


def post_test():
    """
    Runs outside of logging. Used to return positon, etc before finishing test
    """
    # Save logging data with timestamp
    # Note: results are save to /jimstron_data/results which is a mounted volume within docker
    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_T%H%M%S")
    # logging_save_(f"jimstron_data_{timestamp}")
    pass

##################################################################################
#
##################################################################################

if __name__ == "__main__":
    try:
        rospy.sleep(0.5)  # Wait for node subscribers to subscribe 

        rospy.loginfo("Test Started!")
        test_status_pub_.publish("PRE-TEST")
        pre_test()
  
        test_status_pub_.publish("RUNNING") 
        rospy.wait_for_service('logger/start') # Should be running but to make sure
        logging_start_()
        test()
        logging_stop_()
        
        test_status_pub_.publish("POST-TEST")
        post_test()

        if (test_stop_flag): rospy.logwarn("Test stopped by user")
        test_status_pub_.publish("COMPLETE")
        rospy.loginfo("Test Complete!")

    except rospy.ROSInterruptException:
        rospy.logerror("Test Failed!")
