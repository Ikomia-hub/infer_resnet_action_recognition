from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from infer_resnet_action_recognition.infer_resnet_action_recognition_process import ResNetActionRecognitionFactory
        # Instantiate process object
        return ResNetActionRecognitionFactory()

    def getWidgetFactory(self):
        from infer_resnet_action_recognition.infer_resnet_action_recognition_widget import ResNetActionRecognitionWidgetFactory
        # Instantiate associated widget object
        return ResNetActionRecognitionWidgetFactory()
