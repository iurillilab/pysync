from pathlib import Path

import numpy as np
import pandas as pd
from pynapple import TsdFrame

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


class TreadmillData(TsdFrame):
    """Class to handle treadmill data."""

    FILE_PATTERN = "{}_data.csv"
    POST_HOC_CALIBRATION = BALL_CALIBRATION

    # def __init__(self):

    #     epochs_df = self.stim_epochs_df

    #     s = ~np.isnan(epochs_df.barcode_cpu_time)
    #     # cpu_base = TimeBase(epochs_df.barcode_cpu_time[s])
    #     intan_base = TimeBase(epochs_df.barcode_intan_time[s])

    @classmethod
    def from_folder(cls, path):
        path = Path(path)
        csv_datasets = list(path.glob(TreadmillData.FILE_PATTERN.format("*")))
        assert len(csv_datasets) > 0, f"No treadmill data found in {path}"

        # TODO change once Tsd concatenation has been solved
        #df_list = []
        #for csv_dataset in csv_datasets:
        #    df_list.append(cls.from_csv(csv_dataset))
        df_list = []
        for csv_dataset in csv_datasets:
            df_list.append(pd.read_csv(csv_dataset))

        df = pd.concat(df_list).reset_index()
        df = TreadmillData.preprocess_df(df)

        return cls(d=df.values, t=df.index.values, columns=df.columns)

    @classmethod
    def from_csv(cls, file_path):
        df = pd.read_csv(file_path)

        df = TreadmillData.preprocess_df(df)

        return cls(df)

    @staticmethod
    def preprocess_df(df):
        df.loc[:, ["pitch", "roll", "yaw"]] = (
            TreadmillData.POST_HOC_CALIBRATION @ df[["x0", "y0", "x1", "y1"]].values.T
        ).T

        # Convert timecolumn with nanosecond timestamps in seconds timedeltas:
        t_s = pd.to_datetime(df["t_ns"], unit="ns")
        t_s -= t_s[0]

        df["time"] = t_s.dt.total_seconds()

        df = df.set_index("time")

        return df


if __name__ == "__main__":
    from pathlib import Path
    data_folder = Path("/Users/vigji/Desktop/batch6_ephys_data/M9")
    treadmill_data = TreadmillData.from_folder(data_folder / "TreadmillData")
