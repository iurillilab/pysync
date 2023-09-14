import numpy as np


# This matrix has been manually calibrated from some raw data (known number of rotations by hand)

# Pitch, roll, yaw
true_rot = 8 * np.pi
BALL_CALIBRATION = np.array(
    [
        [true_rot / 120000,                    0.0,                0.0,                   0.0],
        [              0.0,                    0.0,  true_rot / 500000,                   0.0],
        [              0.0,  true_rot / (210000*2),                0.0,  true_rot / (630000*2)],
    ])

class TreadmillData:
    pass
