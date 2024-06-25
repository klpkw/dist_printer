# This represents a detected or tracked object with reference coordinate frame and timestamp.

std_msgs/Header header

# The id of the object (presumably from the detecting sensor).
uint32 id

# The detected position and orientation of the object.
geometry_msgs/Pose pose

# The dimensions of the object
geometry_msgs/Vector3 dimensions

# The detected linear and angular velocities of the object.
geometry_msgs/Twist twist

# The detected linear and angular accelerations of the object.
geometry_msgs/Accel accel

# Predicted path of the object
nav_msgs/Path predicted_path

# The type of classification given to this object.
uint8 classification
uint8 CLASSIFICATION_PEDESTRIAN=0
uint8 CLASSIFICATION_CAR=1
uint8 CLASSIFICATION_TRUCK=2
uint8 CLASSIFICATION_BUS=3
uint8 CLASSIFICATION_TRAILER=4
uint8 CLASSIFICATION_MOTORCYCLE=5
uint8 CLASSIFICATION_BICYCLE=6
uint8 CLASSIFICATION_UKNOWN=99

# The number of scans/frames from the sensor that this object has been classified as the current classification.
uint32 classification_age
