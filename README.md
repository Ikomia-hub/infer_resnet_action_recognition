<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/infer_resnet_action_recognition/main/icon/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_resnet_action_recognition</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_resnet_action_recognition">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_resnet_action_recognition">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_resnet_action_recognition/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_resnet_action_recognition.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>


Run ResNets on videos for action recognition.

![Kinetics illustration](https://production-media.paperswithcode.com/datasets/kinetics.jpg)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow


```python
from ikomia.utils.displayIO import display
from ikomia.core import IODataType
from ikomia.dataprocess import CImageIO
from ikomia.dataprocess.workflow import Workflow
import cv2

# Init your workflow
wf = Workflow()

# Add object detection algorithm
detector = wf.add_task(name="infer_resnet_action_recognition", auto_connect=True)

stream = cv2.VideoCapture(0)
while True:
    # Read image from stream
    ret, frame = stream.read()

    # Test if streaming is OK
    if not ret:
        continue

    # Run the workflow on current frame
    # We don't run at workflow level as action recognition algorithm need to accumulate frames
    # and frame stack is cleared when a workflow is started 
    detector.set_input(CImageIO(IODataType.IMAGE, frame), 0)
    detector.run()

    # Get results
    image_out = detector.get_output(0)
    graphics_out = detector.get_output(1)

    # Convert color space
    img_res = cv2.cvtColor(image_out.get_image_with_graphics(graphics_out), cv2.COLOR_BGR2RGB)

    # Display using OpenCV
    display(img_res, title="Action recognition", viewer="opencv")

    # Press 'q' to quit the streaming process
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the stream object
stream.release()
# Destroy all windows
cv2.destroyAllWindows()
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

- **model_name** (str) - default 'resnet-18-kinetics': Name of the pre-trained model. Additional ResNet size are available: 
    - resnet-34-kinetics
    - resnet-50-kinetics
    - resnet-101-kinetics
    - resnext-101-kinetics.onnx
    - wideresnet-50-kinetics.onnx

- **rolling** (bool) - default 'True': Number of frame passed has input. 
- **sample_duration** (int) - default '16': Number of frame passed as input. 

If rolling frame prediction is **used**, we perform N classifications, one for each frame (once the deque data structure is filled, of course)
If rolling frame prediction is **not used**, we only have to perform N / SAMPLE_DURATION classifications, thus reducing the amount of time it takes to process a video stream significantly.


**Parameters** should be in **strings format**  when added to the dictionary.


```python
from ikomia.utils.displayIO import display
from ikomia.core import IODataType
from ikomia.dataprocess import CImageIO
from ikomia.dataprocess.workflow import Workflow
import cv2

# Init your workflow
wf = Workflow()

# Add object detection algorithm
detector = wf.add_task(name="infer_resnet_action_recognition", auto_connect=True)
detector.set_parameters({
    "model_name": "resnet-34-kinetics",
    "rolling": "False",
    "sample_duration": "16"
})

stream = cv2.VideoCapture(0)
while True:
    # Read image from stream
    ret, frame = stream.read()

    # Test if streaming is OK
    if not ret:
        continue

    # Run the workflow on current frame
    # We don't run at workflow level as action recognition algorithm need to accumulate frames
    # and frame stack is cleared when a workflow is started 
    detector.set_input(CImageIO(IODataType.IMAGE, frame), 0)
    detector.run()

    # Get results
    image_out = detector.get_output(0)
    graphics_out = detector.get_output(1)

    # Convert color space
    img_res = cv2.cvtColor(image_out.get_image_with_graphics(graphics_out), cv2.COLOR_BGR2RGB)

    # Display using OpenCV
    display(img_res, title="Action recognition", viewer="opencv")

    # Press 'q' to quit the streaming process
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the stream object
stream.release()
# Destroy all windows
cv2.destroyAllWindows()
```
