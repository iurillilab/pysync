import pynapple as nap
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import numpy as np
import functools    


class SpikeData:
    FILE_CACHE_NAME = "spikes_df.npy"

    def __init__(self, folder_path: Path, force_loading: bool = False):
        self.root = Path(folder_path)
        cache_filename = self.root / self.FILE_CACHE_NAME 
        
        if cache_filename.exists() or force_loading:
            spikes_vals = np.load(cache_filename)
            spikes_df = pd.DataFrame(spikes_vals, columns=["time", "nid"])
            spikes_df["nid"] = spikes_df["nid"].astype(int)
        else:
            df_list = []
            for i, spks in self.pynapple_data.items():
                neuron_df = pd.DataFrame(dict(time=spks.index, nid=i))
                df_list.append(neuron_df)

            spikes_df = pd.concat(df_list)
            spikes_df = spikes_df.sort_values(by="time")
            spikes_df = spikes_df.reset_index(drop=True)
            np.save(cache_filename, spikes_df.values)
        
        self.spikes_df = spikes_df

    @functools.cached_property
    def pynapple_data(self):
        return nap.load_folder(str(self.root))["pynapplenwb"]["SpikeData"]["units"]

    @functools.cached_property
    def fr_mat(self):
        n_pts = len(self.npx_barcodes.array)
        n_units = len(self.spikes.index)

        fname = (self.spikes_data_folder / "_custom_fr.npy")
        if fname.exists():
            return np.load(fname)
        else:
            original_fs = 30000
            new_sampling = 200
            spike_traces = np.zeros((n_pts // (original_fs // new_sampling), n_units))

            for idx, i in tqdm(enumerate(self.spikes.index)):
                spike_traces[(self.spikes[i].index.values[:-2] * new_sampling).astype(int), idx] = 1

            spike_traces = gaussian_filter(spike_traces, (4, 0))
            spike_traces = spike_traces[::2, :]

            np.save(fname, spike_traces)

            return spike_traces
        
    @functools.cached_property
    def neurons_idxs(self):
        return self.spikes_df["nid"].unique()
    
    @property
    def n_neurons(self):
        return len(self.neurons_idxs)


    @functools.cached_property
    def units_depth_df(self):
        temps = np.load(self.spikes_data_folder / "templates.npy")
        chan_idxs = temps.mean(1).argmax(1)
        channel_positions = np.load(self.spikes_data_folder / "channel_positions.npy")
        units_positions = channel_positions[chan_idxs, :]
        good_idxs = self.spikes.index
        df = pd.DataFrame(units_positions[good_idxs, [1]], index=good_idxs, columns=["depth"])
        df["tot_spikes"] = self.spikes_df.groupby("nid").count()
        df["nid"] = [f"{self.root.name}_{i}" for i in df.index]
        df["mid"] = self.root.name

        POS_DICT = dict(m7=(0, 1000, 3000, 4000),
                        tofixm6=(0, 700, 2500, 4000),
                        m5=(0, 1070, 2400, 4000),
                        m4=(0, 1550, 2940, 4000),
                        m8=(0, 1300, 2700, 4000))

        if self.root.name.lower() in POS_DICT.keys():
            depth_lims = POS_DICT[self.root.name.lower()]
        else:
            depth_lims = 0, 1000, 3000, 4000

        df["area"] = 0
        df.loc[df["depth"] > depth_lims[-2], "area"] = "ctx"
        df.loc[(df["depth"] > depth_lims[1]) & (df["depth"] < depth_lims[-2]), "area"] = "sc"
        df.loc[df["depth"] < depth_lims[1], "area"] = "mid"

        return df
    

if __name__=="__main__":
    from datetime import datetime

    t = datetime.now()
    spikedata = SpikeData("/Users/vigji/Desktop/batch6_ephys_data/M9/SpikeData")

    print((datetime.now() - t).total_seconds())
    t = datetime.now()
    spikedata.units_depth_df
    print((datetime.now() - t).total_seconds())