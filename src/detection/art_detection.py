import rospy
import numpy as np
import cv_bridge
import cv2
from art_detection.msg import ArtResult, ArtResults, CameraPoint
from sensor_msgs.msg import Image
from libart import art

class ArtDetection:
    def __init__(self, imageTopic):

        self.bridge = cv_bridge.CvBridge()

        # ## Setup publishers
        self.publishers = {
            "image_with_tags": rospy.Publisher('art/image_with_tags', Image, queue_size=1),
            "tags": rospy.Publisher('/art/tags', ArtResults)
        }
        # ## Setup subscribers
        self.subscribers = {
            "image" : rospy.Subscriber(imageTopic, Image, self._onImageRecieved)
        }
        
        self.currentImage = None
        self.runs_since_image = 0 
    
    def _onImageRecieved(self, msg):
        self.currentImage = msg
    
    # Takes an image, runs it through detections
    def _processImage(self):
        data = self.currentImage
        if(data is None):
            # No image.
            self.runs_since_image += 1
            if (self.runs_since_image >= 60): 
                rospy.logwarn("Waiting for image...")
                self.runs_since_image = 0
            return
        self.runs_since_image = 0
        
        # Get Image and pre-process
        image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8') # Makes the ROS image work with pyTorch
        
        results = art.findTags(image)

        # Publish the result
        self._publishImage(results)

    def _publishImage(self, data):
        image = self.currentImage

        for tagValue, corners in data:
            cv2.polylines(image, [corners], True, (50,0,200), 2)
            cv2.putText(image, f"{tagValue}", corners[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        self.publishers["image_with_tags"].publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))

    def _publishTags(self, data):
        output = ArtResults()
        output.resultsCount = len(data)
        
        res = []
        for tagID, corners in data:
            tag = ArtResult()
            tag.tagID = tagID
            tag.corners = [CameraPoint(point[0], point[1]) for point in corners]
            res.append(tag)

        output.results = res

        self.publishers["tags"].publish(output)
    
    @staticmethod
    def main(*args, rate, **kwargs):
        rospy.init_node('art') 
        d = ArtDetection(*args, **kwargs)
        rate = rospy.Rate(rate)
        while not rospy.is_shutdown(): # Main loop.
            d._processImage()
            rate.sleep()
