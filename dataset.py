
from roboflow import Roboflow
rf = Roboflow(api_key="n602tVBvomRUnBWeFyLG")
project = rf.workspace("work-sxiv5").project("robocon2025-hoop-detection-2fr3q")
version = project.version(3)
dataset = version.download("yolov8")
