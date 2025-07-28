
from roboflow import Roboflow

rf = Roboflow(api_key="n602tVBvomRUnBWeFyLG")
project = rf.workspace("work-sxiv5").project("basketball-f2rlv")
version = project.version(1)
dataset = version.download("yolov8")
