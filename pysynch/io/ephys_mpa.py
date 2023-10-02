import numpy as np
from pathlib import Path
import pandas as pd
import functools
import ast
from pysynch.core import TimeBaseTs, DigitalTsd
from pysynch.io import PsychoScriptsLogFolder, ECEphysExpIntanData, TreadmillData, PsychoScriptsLog
from pysynch.io.spike_data import SpikeData
from tqdm import trange, tqdm

fig_folder = Path(r"/Users/vigji/Desktop/one2one_figs")

class MPASCExperiment:
    """DatasetLoader class for an electrophysiology dataset of the MPA series.
    Timebase is assumed to be the spike data timebase, all other events are translated to this one.

    """

    BARCODE_LEN = 6.43
    NPX_FS = 30000

    def __init__(self, path):
        self.root = Path(path)

        self.intan_data_folder = self.root / "IntanData"
        self.video_data_folder = self.root / "Video"
        self.npx_data_folder = self.root / "NPXData"
        self.spikes_data_folder = self.root / "SpikeData"
        self.treadmill_data_folder = self.root / "TreadmillData"
        self.events_data_folder = self.root / "EventData"

        self._spikes_df = None
        self.spikes_fs = MPASCExperiment.NPX_FS
        
        # Exclude some stimulus presentations based on file:
        if (self.events_data_folder / "toexclude.txt").exists():
            self.to_exclude = open(self.events_data_folder / "toexclude.txt").read().splitlines()
        else:
            self.to_exclude = []

    @functools.cached_property
    def spike_data(self):
        return SpikeData(self.spikes_data_folder)
            
    @functools.cached_property
    def treadmill_data(self):
        treadmill_data = TreadmillData.from_folder(self.treadmill_data_folder)
        treadmill_data = pd.DataFrame(index=treadmill_data.index, data=treadmill_data.values, columns=treadmill_data.columns)
        
        epochs_df = self.stim_epochs_df
        s = ~np.isnan(epochs_df.barcode_cpu_time)
        cpu_base = TimeBaseTs(epochs_df.barcode_cpu_time[s].values)
        intan_base = TimeBaseTs(epochs_df.barcode_intan_time[s].values)

        treadmill_data["t_intan"] = cpu_base.map_times_to(intan_base, treadmill_data["t_ns"])
        
        return treadmill_data
        
    @functools.cached_property
    def stim_data(self):
        return PsychoScriptsLogFolder(self.events_data_folder)
        
    @functools.cached_property
    def intan_data(self):
        return ECEphysExpIntanData.from_folder(self.intan_data_folder)
    
    @functools.cached_property
    def stim_epochs_df(self):
        # match threadmill data on intan data via the cpu timebase and the psychoscripts log:
        epochs_df = self.stim_data.stim_epochs_df
        epochs_df["barcode_intan_time"] = np.nan
        epochs_df.loc[epochs_df["isvisual"], "barcode_intan_time"] = self.intan_data.photodiode_barcode_times
        
        return epochs_df
    
    @functools.cached_property
    def photodiode_sig(self):
        return self.intan_data.loc["photodiode"]
    
    @functools.cached_property
    def laser_sig(self):
        return self.intan_data.loc["laser"]
    

    @functools.cached_property
    def servo_loom_stim_df(self):
        # onset_times = find_screen_barcode_onsets_times(self.photodiode_sig)
        
        visual_epochs = self.stim_epochs_df[self.stim_epochs_df["isvisual"]]
        onset_rf_idxs = visual_epochs[visual_epochs["stimulus"] == "servo_visual_looming"].index[0]

        # Start and end times:
        start_t, end_t = visual_epochs.loc[onset_rf_idxs, "barcode_intan_time"] + self.BARCODE_LEN, \
                                            visual_epochs.loc[onset_rf_idxs + 1, "barcode_intan_time"]

        # Calculate visual trials start onsets and end onsets:
        photodiode_onsets = self.photodiode_sig.onset_times.index
        photodiode_onsets = photodiode_onsets[(start_t < photodiode_onsets) & (photodiode_onsets < end_t)]
        photodiode_starts, photodiode_ends = photodiode_onsets[::2], photodiode_onsets[1::2]

        # Laser trials start and end:
        laser_onsets = self.laser_sig.onset_times.index
        laser_onsets = laser_onsets[(start_t < laser_onsets) & (laser_onsets < end_t)]
        laser_starts = laser_onsets[np.insert(np.diff(laser_onsets) > 1, 0, True)]  # filter for trials
        laser_ends = laser_onsets[
            np.insert(np.diff(laser_onsets) > 1, len(laser_onsets) - 1, True)]  # filter for trials

        # Servo trials start and end:
        servo_onsets = self.servo_sig.onset_times.index
        servo_onsets = servo_onsets[(start_t < servo_onsets) & (servo_onsets < end_t)]
        servo_starts, servo_ends = servo_onsets[::2], servo_onsets[1::2]

        trials_df = []

        lines = PsychoScriptsLog(visual_epochs.loc[onset_rf_idxs, "filename"]).all_lines
        for l in lines:
            if "\tEXP \tTrial: {" in l:
                line_dict = ast.literal_eval(l.split("\t")[2][7:])
                line_dict["time"] = float(l.split("\t")[0])
                line_dict["time_ns"] = int(l.split("\t")[-1][2:-2])
                trials_df.append(line_dict)

        trials_df = pd.DataFrame(trials_df)

        for colname in ["start", "end", "laser_start", "laser_end"]:
            trials_df[colname] = np.nan

        
        # In some experiments for an arduino bug the laser is on at the beginning of the script.
        # we exclude the first pulse, and the last one if it was the one of the next stimulus
        laser_pulses_slice = slice(1, sum(trials_df["laser"]) + 1)
        trials_df.loc[trials_df["laser"], "laser_start"] = laser_starts[laser_pulses_slice]
        trials_df.loc[trials_df["laser"], "laser_end"] = laser_ends[laser_pulses_slice]

        trials_df.loc[trials_df["trial_type"] == "visual", "start"] = photodiode_starts
        trials_df.loc[trials_df["trial_type"] == "visual", "end"] = photodiode_ends

        trials_df.loc[trials_df["trial_type"] == "servo", "start"] = servo_starts
        trials_df.loc[trials_df["trial_type"] == "servo", "end"] = servo_ends

        trials_df.loc[trials_df["trial_type"] == "laser_only", "start"] = \
            trials_df.loc[trials_df["trial_type"] == "laser_only", "laser_start"]
        trials_df.loc[trials_df["trial_type"] == "laser_only", "end"] = \
            trials_df.loc[trials_df["trial_type"] == "laser_only", "laser_end"]

        #for key_to_conv in ["start", "end", "laser_start", "laser_end"]:
        #    trials_df[key_to_conv + "_npx"] = self.intan_to_npx_times(trials_df[key_to_conv])

        return trials_df
    
    @functools.cached_property
    def servo_sig(self):
        return DigitalTsd(self.intan_data.loc["servo_on"].values | self.intan_data.loc["servo_off"].values,
                          self.intan_data.index)
    
    def check_camera_idxs(self):
        DEVIATION_FACTOR = 3  # suspicious interfame times are DEVIATION_FACTOR * median_time
        cam_onsets = self.intan_data.loc["camera"].onset_idxs
        inter_frame_time = np.diff(cam_onsets)
        median_dist = np.median(inter_frame_time) 
        suspicious_inter_frames = np.argwhere(inter_frame_time > median_dist * DEVIATION_FACTOR)

        if len(suspicious_inter_frames) > 0:
            raise ValueError("Implement a solution to resolve non-continuous frame aquisition!")
    
    @property
    def top_camera_trigger_idxs(self):
        return self.intan_data.loc["camera"].onset_idxs.index
    
    @property
    def top_camera_trigger_times(self):
        return self.intan_data.loc["camera"].onset_times.index
    
    def get_closer_frame_to_time(self, time, timebase="intan"):
        if timebase == "intan":
            return np.argmin(np.abs(self.top_camera_trigger_times - time))
        else:
            raise ValueError("only intan implemented!")
        
    def _get_video_frames(self, video_sourcefile, frame_idxs, verbose=True):
        import cv2
        # Open the video file
        cap = cv2.VideoCapture(str(video_sourcefile))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = np.zeros((len(frame_idxs), frame_height, frame_width), dtype=np.uint8)
        # Read the specified frames
        if verbose:
            _wrap = tqdm
        else:
            _wrap = lambda x: x
        for i, frame_number in _wrap(enumerate(frame_idxs)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret:
                frames[i, :, :] = frame[:, :, 0]
        # Release the video capture object
        cap.release()

        return frames

    def get_eye_video_frames(self):
        pass

    def get_topview_video_frames(self, frame_idxs):
        file = next((self.video_data_folder / "top-camera").glob("Basler*.mp4"))

        return self._get_video_frames(file, frame_idxs=frame_idxs)



if __name__ == "__main__":
    exp = MPASCExperiment(Path('/Users/vigji/Desktop/synched_dir/M11'))
    from datetime import datetime

    now = datetime.now()
    exp.get_topview_video_frames(np.arange(100, 100))
    print((datetime.now() - now).total_seconds())

    
