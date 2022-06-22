#!/usr/bin/env python3
import os
import sys
import copy
import re
import importlib
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from rclpy.exceptions import ParameterNotDeclaredException
from rcl_interfaces.msg import Parameter
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.msg import ParameterDescriptor
import sensor_msgs.msg
import std_msgs.msg
from tflite_msgs.msg import TFLite, TFInference
import cv2
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile
import tflite_runtime.interpreter as tflite
from PIL import Image


if cv2.__version__ < "4.0.0":
    raise ImportError("Requires opencv >= 4.0, "
                      "but found {:s}".format(cv2.__version__))

class NXPTFLite(Node):

    def __init__(self):

        super().__init__("nxp_tf_lite_node")

        # Get paramaters or defaults
        model_file_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING,
            description='.tflite model to be executed.')
        label_file_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING,
            description='Name of file containing labels.')
        ext_delegate_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING,
            description='external_delegate_library path.')
        ext_delegate_options_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING,
            description='external delegate options, format: "option1: value1; option2: value2"')
        input_mean_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,
            description='Input_mean.')
        input_std_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,
            description='Input standard deviation.')
        num_threads_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_INTEGER,
            description='Number of threads.')
        camera_image_topic_0_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING,
            description='Camera 0 image topic.')
        output_topic_0_string_array_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING,
            description='Topic 0 name to publish inferences.')
        camera_image_topic_1_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING,
            description='Camera 1 image topic.')
        output_topic_1_string_array_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING,
            description='Topic 1 name to publish inferences.')
        use_gpu_inference_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_BOOL,
            description='Use GPU for inference boolean')
        bbox_index_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_INTEGER,
            description='Index for bbox in model output, -1 if not in model.')
        score_index_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_INTEGER,
            description='Index for confidence score in model output.')
        class_index_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_INTEGER,
            description='Index for class in model output.')

        self.declare_parameter("model_file", "mobilenet_ssd_v2_coco_quant_postprocess.tflite", 
            model_file_descriptor)
        self.declare_parameter("label_file", "coco.txt", 
            label_file_descriptor)
        self.declare_parameter("ext_delegate", "/usr/lib/libvx_delegate.so", 
            ext_delegate_descriptor)
        self.declare_parameter("ext_delegate_options", "", 
            ext_delegate_options_descriptor)
        self.declare_parameter("input_mean", 127.5, 
            input_mean_descriptor)
        self.declare_parameter("input_std", 127.5, 
            input_std_descriptor)
        self.declare_parameter("num_threads", 0,
            num_threads_descriptor)
        self.declare_parameter("camera_image_0", "/NPU/image_raw", 
            camera_image_topic_0_descriptor)
        self.declare_parameter("camera_image_1", "/NPU/image_sim", 
            camera_image_topic_1_descriptor)
        self.declare_parameter("topic_name_0", "TFLiteReal", 
            output_topic_0_string_array_descriptor)
        self.declare_parameter("topic_name_1", "TFLiteSim", 
            output_topic_1_string_array_descriptor)
        self.declare_parameter("gpu_inference", False, 
            use_gpu_inference_descriptor)
        self.declare_parameter("bbox_index", 0, 
            bbox_index_descriptor)
        self.declare_parameter("score_index", 2, 
            score_index_descriptor)
        self.declare_parameter("class_index", 1, 
            class_index_descriptor)

        self.modelFile = self.get_parameter("model_file").value
        self.labelFile = self.get_parameter("label_file").value
        self.extDelegate = self.get_parameter("ext_delegate").value
        self.extDelegateOptions = self.get_parameter("ext_delegate_options").value
        self.inputMean = self.get_parameter("input_mean").value
        self.inputStd = self.get_parameter("input_std").value
        self.numberThreads = self.get_parameter("num_threads").value
        self.cameraImageTopic0 = self.get_parameter("camera_image_0").value
        self.inferenceTopicName0 = self.get_parameter("topic_name_0").value
        self.cameraImageTopic1 = self.get_parameter("camera_image_1").value
        self.inferenceTopicName1 = self.get_parameter("topic_name_1").value
        self.useGPUInference = self.get_parameter("gpu_inference").value
        self.boundingBoxIndex = int(self.get_parameter("bbox_index").value)
        self.classIndex = int(self.get_parameter("class_index").value)
        self.scoreIndex = int(self.get_parameter("score_index").value)

        self.modelsPath = os.path.realpath(os.path.relpath(os.path.join(os.path.realpath(__file__).replace("nxp_tf_lite_node.py",""),"../models")))

        if self.numberThreads < 1:
            self.numberThreads = None
        self.inferenceInitialized = False
        self.bridge = CvBridge()

        if self.useGPUInference:
            os.environ["USE_GPU_INFERENCE"]="1"
        else:
            os.environ["USE_GPU_INFERENCE"]="0"
        
        #Subscribers
        self.imageSub0 = self.create_subscription(sensor_msgs.msg.Image, '{:s}'.format(self.cameraImageTopic0), self.imageCallback0, qos_profile_sensor_data)
        self.imageSub1 = self.create_subscription(sensor_msgs.msg.Image, '{:s}'.format(self.cameraImageTopic1), self.imageCallback1, qos_profile_sensor_data)

        #Publishers
        self.TFLitePub0 = self.create_publisher(TFLite,'{:s}'.format(self.inferenceTopicName0), 0)
        self.TFLitePub1 = self.create_publisher(TFLite,'{:s}'.format(self.inferenceTopicName1), 0)

        with open(os.path.join(self.modelsPath, self.labelFile), 'r') as f:
            self.labels=[line.strip() for line in f.readlines()]

        extDelegate = None
        extDelegateOptions = {}

        if self.extDelegateOptions is not None:
            options = self.extDelegateOptions.split(';')
            for o in options:
                kv = o.split(':')
                if(len(kv) == 2):
                    extDelegateOptions[kv[0].strip()] = kv[1].strip()

        if self.extDelegate is not None:
            print("Loading external delegate from {} with args: {}".format(self.extDelegate, extDelegateOptions))
            extDelegate = [ tflite.load_delegate(self.extDelegate, extDelegateOptions) ]

        self.interpreter = tflite.Interpreter(
            model_path=os.path.join(self.modelsPath, self.modelFile), experimental_delegates=extDelegate, num_threads=self.numberThreads)
        self.interpreter.allocate_tensors()

        self.inputDetails = self.interpreter.get_input_details()
        self.outputDetails = self.interpreter.get_output_details()

        self.floatingModel = self.inputDetails[0]['dtype'] == np.float32

        self.modelHeight = self.inputDetails[0]['shape'][1]
        self.modelWidth = self.inputDetails[0]['shape'][2]


    def loadTFLite(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (frame.shape[0] != self.modelHeight) or (frame.shape[1] != self.modelWidth):
            frame = cv2.resize(frame, (self.modelWidth, self.modelHeight), interpolation = cv2.INTER_AREA)
        img = Image.fromarray(frame)

        inputData = np.expand_dims(img, axis=0)

        if self.floatingModel:
            inputData = (np.float32(inputData) - self.inputMean) / self.inputStd

        self.interpreter.set_tensor(self.inputDetails[0]['index'], inputData)

        # ignore the 1st invoke
        startTime = self.get_clock().now().nanoseconds
        self.interpreter.invoke()
        delta = float((self.get_clock().now().nanoseconds - startTime)/1000000.0)
        infoString='Warm-up time: {:f} ms'.format(delta)
        self.get_logger().info(infoString)
        print(infoString)
        self.inferenceInitialized = True

    def runTFLite(self, frame, image_header, topic):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (frame.shape[0] != self.modelHeight) or (frame.shape[1] != self.modelWidth):
            frame = cv2.resize(frame, (self.modelWidth, self.modelHeight), interpolation = cv2.INTER_AREA)
        img = Image.fromarray(frame)

        inputData = np.expand_dims(img, axis=0)

        if self.floatingModel:
            inputData = (np.float32(inputData) - self.inputMean) / self.inputStd

        self.interpreter.set_tensor(self.inputDetails[0]['index'], inputData)

        startTime = self.get_clock().now().nanoseconds
        self.interpreter.invoke()
        delta = float((self.get_clock().now().nanoseconds - startTime)/1000000.0)

        
        if self.classIndex >= 0:
            inferredClassIndexs = np.squeeze(self.interpreter.get_tensor(self.outputDetails[self.classIndex]['index'])).astype(int)
            inferredLabels = np.take(self.labels, inferredClassIndexs)
        if self.scoreIndex >= 0:
            inferredScores = np.squeeze(self.interpreter.get_tensor(self.outputDetails[self.scoreIndex]['index']))
        if self.boundingBoxIndex >= 0:
            inferredBoundingBoxs = np.squeeze(self.interpreter.get_tensor(self.outputDetails[self.boundingBoxIndex]['index'])).astype(float)
        else:
            inferredBoundingBoxs = np.zeros((len(inferredScores),4), dtype=float)
        
        msgTFLite = TFLite()
        msgTFLite.header.stamp = self.get_clock().now().to_msg()
        msgTFLite.camera_info = image_header
        msgTFLite.inference_time_ms = delta
        inferenceArray=[]
        for ret in range(len(inferredLabels)):
            msgTFInference = TFInference()
            msgTFInference.label=str(inferredLabels[ret])
            msgTFInference.score=float(inferredScores[ret])
            msgTFInference.bbox=inferredBoundingBoxs[ret]
            inferenceArray.append(msgTFInference)
        msgTFLite.inference=inferenceArray

        if int(topic) == 0:
            self.TFLitePub0.publish(msgTFLite)
        if int(topic) == 1:
            self.TFLitePub1.publish(msgTFLite)

        return

    def imageCallback0(self, data):
        self.imageCallback(data, 0)
        return

    def imageCallback1(self, data):
        self.imageCallback(data, 1)
        return

    def imageCallback(self, data, topic):
        scene = self.bridge.imgmsg_to_cv2(data, "bgr8")
        if not self.inferenceInitialized:
            inferenceReturn = self.loadTFLite(scene)
        else:
            self.runTFLite(scene, data.header, topic)
        return


def main(args=None):
    rclpy.init(args=args)
    node = NXPTFLite()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
