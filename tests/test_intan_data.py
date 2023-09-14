from pathlib import Path
from pysynch.io.intan_data import IntanData

data_path = Path("/Users/vigji/Desktop/batch6_ephys_data/testintandata")

intan_data = IntanData(t=[1,2,3], data=[[0,0,0], [1,1,1]])

