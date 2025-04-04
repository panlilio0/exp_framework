# DevOps Sprint 3 Evaluation Demos

John told me that there were two "proofs" for demonstrating morphological communication.

The first proof involves showing that the actuators' behavior is based on that of the other actuators.
To demonstrate this, we have `run_1.1py` and `run_1.2.py`.
Both of these files train and simulate robots. `run_1.1py` simulates the normal version of the robot. `run_1.2.py` runs a version of the robot with the front 3 actuators "deactivated" (replaced with soft voxels). By running both files, you will see that both robots behave quite differently. You can examine the differences more closely by looking at the plots generated in `score_plots`.

The second proof involves showing that the robot's gait must vary at different speeds. I accomplish this by training and simulating a robot at normal speed, then simulating it again at double speed. This is accomplished by saving the action arrays from the first simulation, and inputting every other one in the second simulation to effectively speed up the robot's actions. To demonstrate morphological communication, the robot should behave and move completely differently the second time (instead of just the first time again but faster). This is performed by `run_2.py`. The relevant plots can be seen in `score_plots_2`.


Credits:
Matthew Meek - Set all of this up.
Thomas Breimer - Made the really good demo I copied and butchered 3 times to set this up.
