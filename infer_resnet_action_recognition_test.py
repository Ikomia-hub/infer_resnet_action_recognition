import logging
import cv2
from ikomia.utils.tests import run_for_test
from ikomia.core import task
import os

logger = logging.getLogger(__name__)


def test(t, data_dict):
    logger.info("===== Test::infer resnet action recognition test =====")
    logger.info("----- Use default parameters")
    img = cv2.imread(data_dict["images"]["detection"]["coco"])
    input_img_0 = t.getInput(0)
    input_img_0.setImage(img)
    params = task.get_parameters(t)
    params["model_name"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "resnet-101_kinetics.onnx")
    return run_for_test(t)
