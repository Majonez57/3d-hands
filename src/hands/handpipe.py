# This file uses google MediaPipe to detect hand landmarks from a 3D Camera
# And publish them onto a ROS topic
import rospy
import cv_bridge
import mediapipe as mp
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, Pose
from visualization_msgs.msg import Marker
import message_filters
import pyrealsense2 as rs2

from art_detection.msg import HandResult3D, HandLandmark3D


# Shortcuts
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# export ROS_MASTER_URI=http://172.22.248.161:11311
# roslaunch realsense2_camera rs_camera.launch align_depth:=true color_width:=424 color_height:=240 color_fps:=60 filters:=pointcloud

class Hands:
    def __init__(self, imageTopic2D: str, depthTopic: str, infoTopic: str):

        self.bridge = cv_bridge.CvBridge()

        self.publishers = {
            "image_with_hands": rospy.Publisher('image_with_hands', Image, queue_size=1),
            "3D_hand" : rospy.Publisher('hand_3d', HandResult3D, queue_size=2)
        }

        image_sub = message_filters.Subscriber(imageTopic2D, Image)
        depth_sub = message_filters.Subscriber(depthTopic, Image)
        info_sub = message_filters.Subscriber(infoTopic, CameraInfo)
        

        ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub, info_sub],2,0.3)
        ts.registerCallback(self._onImages)

        self.currentImage = None
        self.currentDepth = None
        self.intrinsics = None
        self.handModel = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.3)
        
        self.camframe = None
    
    def _onImages(self, imagemsg, depthmsg, cameraInfo):
        im = self.bridge.imgmsg_to_cv2(imagemsg, desired_encoding='bgr8')
        (h,w) = im.shape[:2]
        #im = cv2.resize(im, (w/4, h/4))

        self.currentImage = im

        de = self.bridge.imgmsg_to_cv2(depthmsg, desired_encoding='passthrough')
        (h,w) = de.shape[:2]
        #de = cv2.resize(im, (w/4, h/4))

        self.currentDepth = de
        self.camframe = imagemsg.header.frame_id

        # Intrinsic Matrix
        if self.intrinsics:
            return
        self.intrinsics = rs2.intrinsics()
        self.intrinsics.width = cameraInfo.width
        self.intrinsics.height = cameraInfo.height
        self.intrinsics.ppx = cameraInfo.K[2]
        self.intrinsics.ppy = cameraInfo.K[5]
        self.intrinsics.fx = cameraInfo.K[0]
        self.intrinsics.fy = cameraInfo.K[4]
        if cameraInfo.distortion_model == 'plumb_bob':
            self.intrinsics.model = rs2.distortion.brown_conrady
        elif cameraInfo.distortion_model == 'equidistant':
            self.intrinsics.model = rs2.distortion.kannala_brandt4
        self.intrinsics.coeffs = [i for i in cameraInfo.D]

    def _findHands(self):
        image = self.currentImage
        depth = self.currentDepth

        if (image is None):
            rospy.logwarn("No image detected...")
        
            return #241222073405
        elif depth is None:
            rospy.logwarn("No depth detected...")
            return  
        
        results = self.handModel.process(image)

        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        if not results.multi_hand_landmarks:
            self.publishers["image_with_hands"].publish(self.bridge.cv2_to_imgmsg(annotated_image, "bgr8"))
            return
        handA = []
        handB = []
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # print('hand_landmarks:', hand_landmarks)

            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
                    
            self.publishers["image_with_hands"].publish(self.bridge.cv2_to_imgmsg(annotated_image, "bgr8"))

            hand_markers = [0,1,4,5,8,9,12,13,16,17,20]

            hand_result = HandResult3D()
            hand_result.header.frame_id = self.camframe
            hand_result.header.stamp = rospy.Time.now()

            landmarks = []

            for marker in hand_markers:
                try:
                    tipx = hand_landmarks.landmark[marker].x * image_width
                    tipy = hand_landmarks.landmark[marker].y * image_height 
                    tipdepth = depth[int(tipy), int(tipx)]/1000

                    print(f"index depth: {tipdepth}")

                    # Deprojection!
                    (x,y,z) = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [tipx, tipy], tipdepth)
                

                    point = Point()
                    point.x = z
                    point.y = -x
                    point.z = -y
                    # No orientation for now
                    # The 'facing' direction of the hand may be useful in the future
                    #point.orientation.w = 1

                    if not (z == 0 and x == 0):
                        landmark = HandLandmark3D()
                        landmark.point = point 
                        landmark.landmarkID = marker

                        landmarks.append(landmark)

                        if idx == 0:
                            handA.append(point)
                        else:
                            landmarks.append(landmark)
                            handB.append(point)

                except IndexError: # This is really really lazy
                    # Avoids the index issue when one landmark is out of index
                    continue
                    
            hand_result.landmarks = landmarks

            # point = PointStamped()
            # point.header.stamp = rospy.Time.now()
            # point.header.frame_id = 'camera_link'
            # point.point.x = z
            # point.point.y = -x
            # point.point.z = -y

            self.publishers["3D_hand"].publish(hand_result)
            
            # # Draw hand world landmarks.
            
            # if not results.multi_hand_world_landmarks:
            #     continue
            # for hand_world_landmarks in results.multi_hand_world_landmarks:
            #     mp_drawing.plot_landmarks(hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
    
    @staticmethod
    def main(*args, rate, **kwargs):
        rospy.init_node('hands')
        d = Hands(*args, **kwargs)
        rate = rospy.Rate(rate)
        while not rospy.is_shutdown():
            d._findHands()
            rate.sleep()