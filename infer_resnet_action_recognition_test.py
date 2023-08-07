import logging
import cv2
from ikomia.utils.tests import run_for_test
import os

logger = logging.getLogger(__name__)


def test(t, data_dict):
    logger.info("===== Test::infer resnet action recognition test =====")
    logger.info("----- Use default parameters")
    img = cv2.imread(data_dict["images"]["detection"]["coco"])
    input_img_0 = t.get_input(0)
    input_img_0.set_image(img)
    params = t.get_parameters()
    params["model_path"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "resnext-101-kinetics.onnx")
    t.set_parameters(params)
    return run_for_test(t)
