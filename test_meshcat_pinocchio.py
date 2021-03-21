# This examples shows how to load and move a robot in meshcat.
# Pinocchio can be installed using:
# conda config --add channels conda-forge
# conda install pinocchio
# Note: this feature requires Meshcat to be installed, this can be done using
# ~/anaconda3/.../pip3 install meshcat

import pinocchio as pin
import numpy as np
import sys
import os
from os.path import dirname, join, abspath
import time

from pinocchio.visualize import MeshcatVisualizer

# Load the URDF model.
# Conversion with str seems to be necessary when executing this file with ipython
pinocchio_path = dirname(dirname(str(abspath(__file__)))) + "/pinocchio"
pinocchio_model_dir = join(pinocchio_path, "models")

model_path = join(pinocchio_model_dir, "others/robots")
mesh_dir = model_path
urdf_filename = "talos_reduced.urdf"
urdf_model_path = join(join(model_path, "talos_data/urdf"), urdf_filename)

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
)

viz = MeshcatVisualizer(model, collision_model, visual_model)

# Start a new MeshCat server and client.
# Note: the server can also be started separately using the "meshcat-server" command in a terminal:
# this enables the server to remain active after the current script ends.
#
# Option open=True pens the visualizer.
# Note: the visualizer can also be opened seperately by visiting the provided URL.
try:
    viz.initViewer(open=True)
except ImportError as err:
    print(
        "Error while initializing the viewer. It seems you should install Python meshcat"
    )
    print(err)
    sys.exit(0)

# Load the robot in the viewer.
viz.loadViewerModel()

# Display a robot configuration.
q0 = pin.neutral(model)

loop_counter = 0
step_time = 0.025  # s
end_counter = 5 / step_time
period = 1  # s

while loop_counter < end_counter:
    loop_counter = loop_counter + 1

    q0[7] = np.sin(2 * np.pi * (loop_counter * step_time) / period)
    viz.display(q0)

    time.sleep(step_time)


# # Display another robot.
# viz2 = MeshcatVisualizer(model, collision_model, visual_model)
# viz2.initViewer(viz.viewer)
# viz2.loadViewerModel(rootNodeName="pinocchio2")
# q = q0.copy()
# q[1] = 1.0
# viz2.display(q)
