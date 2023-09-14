from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_resnet_action_recognition.infer_resnet_action_recognition_process import ResNetActionRecognitionParam
import cv2
import os
# PyQt GUI framework
from PyQt5.QtWidgets import *

backend_names = {
    cv2.dnn.DNN_BACKEND_DEFAULT: "Default",
    cv2.dnn.DNN_BACKEND_HALIDE: "Halide",
    cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE: "Inference engine",
    cv2.dnn.DNN_BACKEND_OPENCV: "OpenCV",
    cv2.dnn.DNN_BACKEND_VKCOM: "VKCOM",
    cv2.dnn.DNN_BACKEND_CUDA: "CUDA",
}

target_names = {
    cv2.dnn.DNN_TARGET_CPU: "CPU",
    cv2.dnn.DNN_TARGET_OPENCL: "OpenCL FP32",
    cv2.dnn.DNN_TARGET_OPENCL_FP16: "OpenCL FP16",
    cv2.dnn.DNN_TARGET_MYRIAD: "MYRIAD",
    cv2.dnn.DNN_TARGET_VULKAN: "VULKAN",
    cv2.dnn.DNN_TARGET_FPGA: "FPGA",
    cv2.dnn.DNN_TARGET_CUDA: "CUDA FP32",
    cv2.dnn.DNN_TARGET_CUDA_FP16: "CUDA FP16",
}

backend_targets = {
    cv2.dnn.DNN_BACKEND_DEFAULT: [cv2.dnn.DNN_TARGET_CPU, cv2.dnn.DNN_TARGET_OPENCL, cv2.dnn.DNN_TARGET_OPENCL_FP16],
    cv2.dnn.DNN_BACKEND_HALIDE: [cv2.dnn.DNN_TARGET_CPU],
    cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE: [cv2.dnn.DNN_TARGET_CPU],
    cv2.dnn.DNN_BACKEND_OPENCV: [cv2.dnn.DNN_TARGET_CPU, cv2.dnn.DNN_TARGET_OPENCL, cv2.dnn.DNN_TARGET_OPENCL_FP16],
    cv2.dnn.DNN_BACKEND_VKCOM: [cv2.dnn.DNN_TARGET_CPU],
    cv2.dnn.DNN_BACKEND_CUDA: [cv2.dnn.DNN_TARGET_CUDA, cv2.dnn.DNN_TARGET_CUDA_FP16],
}


# --------------------
# - Class which implements widget associated with the process
# - Inherits core.CProtocolTaskWidget from Ikomia API
# --------------------
class ResNetActionRecognitionWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.param = ResNetActionRecognitionParam()
        else:
            self.param = param

        self.param_changed = False

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        # Sample duration
        label_duration = QLabel("Sample duration (in frame)")
        self.spin_duration = QSpinBox()
        self.spin_duration.setRange(1, 100)
        self.spin_duration.setSingleStep(1)
        self.spin_duration.setValue(self.param.sample_duration)

        # Rolling prediction on/off
        self.check_rolling = QCheckBox("Rolling prediction")
        self.check_rolling.setChecked(self.param.rolling)


        # Combobox for inference backend
        label_backend = QLabel("DNN backend")
        self.combo_backend = QComboBox()
        self.fill_combo_backend() 
        self.combo_backend.setCurrentIndex(self.combo_backend.findData(self.param.backend))
        self.combo_backend.currentIndexChanged.connect(self.on_backend_changed)

        # Combobox for inference target
        label_target = QLabel("DNN target")
        self.combo_target = QComboBox()
        self.fill_combo_target(self.param.backend)
        self.combo_target.setCurrentIndex(self.combo_target.findData(self.param.target))
        self.combo_target.currentIndexChanged.connect(self.on_param_changed)

        # Fill layout       
        self.grid_layout.addWidget(label_backend, 0, 0, 1, 1)
        self.grid_layout.addWidget(self.combo_backend, 0, 1, 1, 1)
        self.grid_layout.addWidget(label_target, 1, 0, 1, 1)
        self.grid_layout.addWidget(self.combo_target, 1, 1, 1, 1)
        # self.grid_layout.addWidget(label_model, 2, 0, 1, 1)
        # self.grid_layout.addWidget(self.combo_models, 2, 1, 1, 1)
        self.grid_layout.addWidget(label_duration, 3, 0, 1, 1)
        self.grid_layout.addWidget(self.spin_duration, 3, 1, 1, 1)
        self.grid_layout.addWidget(self.check_rolling, 4, 0, 1, 2)

        # Combobox for models
        self.combo_models = pyqtutils.append_combo(self.grid_layout, "Model")
        self.combo_models.addItem("resnet-34-kinetics")
        self.combo_models.addItem("resnet-50-kinetics")
        self.combo_models.addItem("resnet-101-kinetics")
        self.combo_models.addItem("resnext-101-kinetics.onnx")
        self.combo_models.addItem("wideresnet-50-kinetics.onnx")
        self.combo_models.setCurrentText(self.param.model_name)

        # PyQt -> Qt wrapping
        layoutPtr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.set_layout(layoutPtr)


    def fill_combo_backend(self):
        self.combo_backend.clear()
        for backend in backend_names:            
            self.combo_backend.addItem(backend_names[backend], backend)

    def fill_combo_target(self, backend):
        targets = backend_targets[backend]
        self.combo_target.clear()

        for target in targets:
            self.combo_target.addItem(target_names[target], target)

    def on_backend_changed(self, index):
        backend = self.combo_backend.currentData()
        self.fill_combo_target(backend)
        self.param.update = True

    def on_param_changed(self, index):
        self.param.update = True

    def on_apply(self):
        # Apply button clicked slot
        # Get parameters from widget
        self.param.update = True
        self.param.sample_duration = self.spin_duration.value()
        self.param.rolling = self.check_rolling.isChecked()
        self.param.model_name = self.combo_models.currentText()
        self.param.update = self.param_changed
        self.param.backend = self.combo_backend.currentData()
        self.param.target = self.combo_target.currentData()

        # Send signal to launch the process
        self.emit_apply(self.param)


# --------------------
# - Factory class to build process widget object
# - Inherits dataprocess.CWidgetFactory from Ikomia API
# --------------------
class ResNetActionRecognitionWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_resnet_action_recognition"

    def create(self, param):
        # Create widget object
        return ResNetActionRecognitionWidget(param, None)
