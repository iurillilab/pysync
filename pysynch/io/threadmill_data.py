import numpy as np

# This matrix has been manually calibrated from some raw data (known number of rotations by hand)

# Pitch, roll, yaw
true_rot = 8 * np.pi
BALL_CALIBRATION = np.array(
    [
        [true_rot / 120000, 0.0, 0.0, 0.0],
        [0.0, 0.0, true_rot / 500000, 0.0],
        [0.0, true_rot / (210000 * 2), 0.0, true_rot / (630000 * 2)],
    ]
)


class TreadmillData:
    def __init__(self):
            
        epochs_df = self.stim_epochs_df

        s = ~np.isnan(epochs_df.barcode_cpu_time)
        cpu_base = TimeBase(epochs_df.barcode_cpu_time[s])
        intan_base = TimeBase(epochs_df.barcode_intan_time[s])

        df = pd.read_csv(next(self.treadmill_data_folder.glob("*.csv")))
        df["t_intan"] = cpu_base.map_times_to(intan_base, df["t_ns"])
        df.loc[:, ["pitch", "roll", "yaw"]] = (
            BALL_CALIBRATION @ df[["x0", "y0", "x1", "y1"]].values.T
        ).T
        return df

    def from_csv(file_path)