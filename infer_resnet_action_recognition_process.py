from ikomia import utils, core, dataprocess
import copy
import os
import cv2
import imutils
import numpy as np
from collections import deque
import requests

SAMPLE_SIZE = 112


# --------------------
# - Class to handle the process parameters
# - Inherits core.CProtocolTaskParam from Ikomia API
# --------------------
class ResNetActionRecognitionParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.model_name = 'resnet-34-kinetics'
        self.rolling = True
        self.sample_duration = 16
        self.update = False
        self.backend = cv2.dnn.DNN_BACKEND_DEFAULT
        self.target = cv2.dnn.DNN_TARGET_CPU

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_name = str(param_map["model_name"])
        self.rolling = bool(param_map["rolling"])
        self.sample_duration = int(param_map["sample_duration"])
        self.update = True
        self.backend = int(param_map["backend"])
        self.target = int(param_map["target"])

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {
            "model_name": str(self.model_name),
            "rolling": str(self.rolling),
            "sample_duration": str(self.sample_duration),
            "backend": str(self.backend),
            "target": str(self.target)
        }
        return param_map


# --------------------
# - Class which implements the process
# - Inherits core.CProtocolTask or derived from Ikomia API
# --------------------
class ResNetActionRecognition(dataprocess.CVideoTask):

    def __init__(self, name, param):
        dataprocess.CVideoTask.__init__(self, name)
        self.add_output(dataprocess.CGraphicsOutput())
        self.net = None
        self.last_label = ""
        self.class_names = []
        self.frames = None
        self.models_folder = os.path.dirname(os.path.realpath(__file__)) + "/models"
        self.model_path = None

        # Create parameters class
        if param is None:
            self.set_param_object(ResNetActionRecognitionParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        # Load class names
        model_folder = os.path.dirname(os.path.realpath(__file__)) + "/models"
        with open(model_folder + "/class_names") as f:
            for row in f:
                self.class_names.append(row[:-1])

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def global_input_changed(self, new_sequence):
        if new_sequence:
            self.frames = None
            self.last_label = ""

    def notify_video_start(self, frame_count):
        self.frames = None
        self.last_label = ""

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()

        # Get parameters :
        param = self.get_param_object()
        if self.frames is None:
            if param.rolling:
                self.frames = deque(maxlen=param.sample_duration)
            else:
                self.frames = []

        # Load the recognition model from disk
        if self.net is None or param.update:
            self.model_path = self.models_folder + f"/{param.model_name}.onnx"
            if not os.path.exists(self.model_path):
                print("Downloading model, please wait...")
                model_url = utils.get_model_hub_url() + f"/{self.name}/{param.model_name}.onnx"
                # self.download(model_url, param.model_weight_file)
                response = requests.get(model_url, stream=True)
                with open(self.model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            self.net = cv2.dnn.readNet(self.model_path)
            self.net.setPreferableBackend(param.backend)
            self.net.setPreferableTarget(param.target)
            param.update = False

        # Get input :
        input_img = self.get_input(0)

        # Get image from input (numpy array):
        src_image = input_img.get_image()
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

            if 0 <= class_index < len(self.class_names):
                self.last_label = self.class_names[class_index]
            else:
                self.last_label = ""

            if not param.rolling:
                self.frames = []

        # Get output :
        self.forward_input_image(0, 0)
        graphics_out = self.get_output(1)
        graphics_out.clear_data()
        graphics_out.add_text(self.last_label, 10, 10)

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits dataprocess.CProcessFactory from Ikomia API
# --------------------
class ResNetActionRecognitionFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_resnet_action_recognition"
        self.info.short_description = "Human action recognition with spatio-temporal 3D CNNs."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Classification"
        self.info.version = "1.4.0"
        self.info.icon_path = "icon/icon.png"
        self.info.authors = "Kensho Hara, Hirokatsu Kataoka, Yutaka Satoh"
        self.info.article = "Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?"
        self.info.journal = "CVPR"
        self.info.year = 2018
        self.info.license = "MIT License"
        self.info.documentation_link = "https://www.pyimagesearch.com/2019/11/25/human-activity-recognition-with-opencv-and-deep-learning/"
        self.info.repository = "https://github.com/kenshohara/3D-ResNets-PyTorch"
        self.info.keywords = "3D,CNN,detection,activity,classification,kinetics"

    def create(self, param=None):
        # Create process object
        return ResNetActionRecognition(self.info.name, param)


