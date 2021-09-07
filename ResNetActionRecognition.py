from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class ResNetActionRecognition(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from ResNetActionRecognition.ResNetActionRecognition_process import ResNetActionRecognitionProcessFactory
        # Instantiate process object
        return ResNetActionRecognitionProcessFactory()

    def getWidgetFactory(self):
        from ResNetActionRecognition.ResNetActionRecognition_widget import ResNetActionRecognitionWidgetFactory
        # Instantiate associated widget object
        return ResNetActionRecognitionWidgetFactory()
