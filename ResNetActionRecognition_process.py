from ikomia import core, dataprocess
import copy
import os
import cv2
import imutils
import numpy as np
from collections import deque

SAMPLE_SIZE = 112


# --------------------
# - Class to handle the process parameters
# - Inherits core.CProtocolTaskParam from Ikomia API
# --------------------
class ResNetActionRecognitionParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.rolling = True
        self.sample_duration = 16
        self.model_path = ""
        self.update = False
        self.backend = cv2.dnn.DNN_BACKEND_DEFAULT
        self.target = cv2.dnn.DNN_TARGET_CPU

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.rolling = bool(param_map["rolling"])
        self.sample_duration = int(param_map["sample_duration"])

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        param_map["rolling"] = str(self.rolling)
        param_map["sample_duration"] = str(self.sample_duration)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits core.CProtocolTask or derived from Ikomia API
# --------------------
class ResNetActionRecognitionProcess(dataprocess.CVideoTask):

    def __init__(self, name, param):
        dataprocess.CVideoTask.__init__(self, name)
        self.net = None
        self.last_label = ""
        self.class_names = []
        self.frames = None

        # Create parameters class
        if param is None:
            self.setParam(ResNetActionRecognitionParam())
        else:
            self.setParam(copy.deepcopy(param))

        # Load class names
        model_folder = os.path.dirname(os.path.realpath(__file__)) + "/models"
        with open(model_folder + "/class_names") as f:
            for row in f:
                self.class_names.append(row[:-1])

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def globalInputChanged(self, new_sequence):
        if new_sequence:
            self.frames = None
            self.last_label = ""

    def notifyVideoStart(self, frame_count):
        self.frames = None
        self.last_label = ""

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        # Get parameters :
        param = self.getParam()

        if self.frames is None:
            if param.rolling:
                self.frames = deque(maxlen=param.sample_duration)
            else:
                self.frames = []

        # Load the recognition model from disk
        if self.net is None or param.update == True:
            self.net = cv2.dnn.readNet(param.model_path)
            self.net.setPreferableBackend(param.backend)
            self.net.setPreferableTarget(param.target)
            param.update = False

        # Get input :
        input_img = self.getInput(0)

        # Get image from input (numpy array):
        src_image = input_img.getImage()
        src_image = imutils.resize(src_image, width=400)
        self.frames.append(src_image)

        if len(self.frames) == param.sample_duration:
            size = (SAMPLE_SIZE, SAMPLE_SIZE)
            mean = (114.7748, 107.7354, 99.4750)
            blob = cv2.dnn.blobFromImages(self.frames, 1.0, size, mean, swapRB=False, crop=True)
            blob = np.transpose(blob, (1, 0, 2, 3))
            blob = np.expand_dims(blob, axis=0)
            self.net.setInput(blob)
            outputs = self.net.forward()
            class_index = np.argmax(outputs)

            if class_index >= 0 and class_index < len(self.class_names):
                self.last_label = self.class_names[class_index]
            else:
                self.last_label = ""

            if not param.rolling:
                self.frames = []

        # Get output :
        self.forwardInputImage(0, 0)
        output_img = self.getOutput(0)
        dst_image = output_img.getImage()

        # draw the predicted activity on the frame
        if self.last_label != "":
            labelSize, baseLine = cv2.getTextSize(self.last_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(dst_image, (0, 0), (labelSize[0] + 20, labelSize[1] + 10), (32, 32, 32), cv2.FILLED)
            cv2.putText(dst_image, self.last_label, (10, int((labelSize[1]+10) / 2) + baseLine), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            output_img.setImage(dst_image)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits dataprocess.CProcessFactory from Ikomia API
# --------------------
class ResNetActionRecognitionProcessFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "ResNet Action Recognition"
        self.info.shortDescription = "Human action recognition with spatio-temporal 3D CNNs."
        self.info.description = "The purpose of this study is to determine whether current video datasets have sufficient data " \
                                "for training very deep convolutional neural networks (CNNs) with spatio-temporalthree-dimensional " \
                                "(3D) kernels. Recently, the performance levels of 3D CNNs in the field of action recognition have " \
                                "improved significantly. However, to date, conventional research has only explored relatively shallow  " \
                                "3D architectures. We examine the architectures of various 3D CNNs from relatively shallow to very deep " \
                                "ones on current video datasets. Based on the results of those experiments, the following conclusions " \
                                "could be obtained: (i) ResNet-18 training resulted in significant overfitting for UCF-101, HMDB-51, " \
                                "and ActivityNet but not for Kinetics. (ii) The Kinetics dataset has sufficient data for training of " \
                                "deep 3D CNNs, and enables training of up to 152 ResNets layers, interestingly similar to 2D ResNets on ImageNet. " \
                                "ResNeXt-101 achieved 78.4% average accuracy on the Kinetics test set. (iii) Kinetics pre-trained simple 3D " \
                                "architectures outperforms complex 2D architectures, and the pretrained ResNeXt-101 achieved 94.5% and 70.2% " \
                                "on UCF-101 and HMDB-51, respectively. The use of 2D CNNs trained on ImageNet has produced significant progress " \
                                "in various tasks in image. We believe that using deep 3D CNNs together with Kinetics will retrace the successful " \
                                "history of 2D CNNs and ImageNet, and stimulate advances in computer vision for videos. The codes and pretrained " \
                                "models used in this study are publicly available."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Classification"
        self.info.version = "1.0.1"
        self.info.iconPath = "icon/icon.png"
        self.info.authors = "Kensho Hara, Hirokatsu Kataoka, Yutaka Satoh"
        self.info.article = "Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?"
        self.info.journal = "CVPR"
        self.info.year = 2018
        self.info.license = "MIT License"
        self.info.documentationLink = "https://www.pyimagesearch.com/2019/11/25/human-activity-recognition-with-opencv-and-deep-learning/"
        self.info.repository = "https://github.com/kenshohara/3D-ResNets-PyTorch"
        self.info.keywords = "3D,CNN,detection,activity,classification,kinetics"

    def create(self, param=None):
        # Create process object
        return ResNetActionRecognitionProcess(self.info.name, param)


