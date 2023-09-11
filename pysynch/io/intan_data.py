import os
import struct
from pysynch.class_utils import lazy_property
import numpy as np
import pandas as pd

from pysynch.core.barcode import BarcodeTsd

class IntanData:
    """Class to load and cache in .npy files data from an Intan recording session.

    To access the data, use the following public properties:

    time_array: np.array of time stamps for each sample
    dig_in_data: pd.DataFrame of data, with shape (num_channels, num_samples)

    """

    def __init__(self, folder_path, dig_channel_names=None) -> None:
        self.folder_path = folder_path
        self.file_list = sorted(list(folder_path.glob("*.rhd")))
        self._preloaded_data = {file.stem: file for file in folder_path.glob("*.npy")}
        self._dig_channel_names = dig_channel_names
        self._raw_data = None
        self._dig_in_array = None
        self._time_array = None
        self._barcodes = None

    @lazy_property
    def raw_data(self) -> list:
        if self._raw_data is None:
            self._raw_data = [load_file(file) for file in self.file_list]
        return self._raw_data

    @property
    def time_array(self) -> np.array:
        if self._time_array is None:
            if "time_array" in self._preloaded_data.keys():
                self._time_array = np.load(self._preloaded_data["time_array"])
            else:
                self._time_array = np.concatenate([d["t_dig"] for d in self.raw_data])
                np.save(self.folder_path / "time_array.npy", self._time_array)

        return self._time_array

    @property
    def dig_in_array(self) -> np.array:
        if self._dig_in_array is None:
            if "dig_in_array" in self._preloaded_data.keys():
                self._dig_in_array = np.load(self._preloaded_data["dig_in_array"])
            else:
                self._dig_in_array = np.concatenate(
                    [d["board_dig_in_data"] for d in self.raw_data], axis=1
                ).T
                np.save(self.folder_path / "dig_in_array.npy", self._dig_in_array)
        return self._dig_in_array

    @property
    def data_df(self) -> pd.DataFrame:
        data = np.concatenate([self.time_array[:, None], self.dig_in_array], axis=1)
        if self._dig_channel_names is None:
            self._dig_channel_names = ["time"] + [
                f"channel_{i}" for i in range(self.dig_in_array.shape[1])
            ]
        col_names = ["time"] + self._dig_channel_names
        return pd.DataFrame(data, columns=col_names)

    @property
    def fs(self) -> int:
        if self._fs is None:
            if "fs" in self._preloaded_data.keys():
                self._fs = np.load(self._preloaded_data["fs"])
            else:
                self._fs = self.raw_data[0]["frequency_parameters"][
                    "board_dig_in_sample_rate"
                ]
                np.save(self.folder_path / "fs.npy", self._fs)
        return self._fs

    @property
    def barcodes(self) -> BarcodeTsd:
        if self._barcodes is None:
            if (
                self._dig_channel_names is not None
                and "barcodes" in self.data_df.columns
            ):
                self._barcodes = BarcodeTsd(self.data_df["barcodes"].values, self.fs)
        return self._barcodes


# Define get_bytes_per_data_block function
def get_bytes_per_data_block(header):
    """Calculates the number of bytes in each 60 or 128 sample datablock."""

    # Each data block contains 60 or 128 amplifier samples.
    bytes_per_block = header["num_samples_per_data_block"] * 4  # timestamp data
    bytes_per_block = (
        bytes_per_block
        + header["num_samples_per_data_block"] * 2 * header["num_amplifier_channels"]
    )

    # Auxiliary inputs are sampled 4x slower than amplifiers
    bytes_per_block = (
        bytes_per_block
        + (header["num_samples_per_data_block"] / 4)
        * 2
        * header["num_aux_input_channels"]
    )

    # Supply voltage is sampled 60 or 128x slower than amplifiers
    bytes_per_block = bytes_per_block + 1 * 2 * header["num_supply_voltage_channels"]

    # Board analog inputs are sampled at same rate as amplifiers
    bytes_per_block = (
        bytes_per_block
        + header["num_samples_per_data_block"] * 2 * header["num_board_adc_channels"]
    )

    # Board digital inputs are sampled at same rate as amplifiers
    if header["num_board_dig_in_channels"] > 0:
        bytes_per_block = bytes_per_block + header["num_samples_per_data_block"] * 2

    # Board digital outputs are sampled at same rate as amplifiers
    if header["num_board_dig_out_channels"] > 0:
        bytes_per_block = bytes_per_block + header["num_samples_per_data_block"] * 2

    # Temp sensor is sampled 60 or 128x slower than amplifiers
    if header["num_temp_sensor_channels"] > 0:
        bytes_per_block = bytes_per_block + 1 * 2 * header["num_temp_sensor_channels"]

    return bytes_per_block


def read_qstring(fid):
    """Read Qt style QString.
    The first 32-bit unsigned number indicates the length of the string (in bytes).
    If this number equals 0xFFFFFFFF, the string is null.
    Strings are stored as unicode.
    """

    (length,) = struct.unpack("<I", fid.read(4))
    if length == int("ffffffff", 16):
        return ""

    if length > (os.fstat(fid.fileno()).st_size - fid.tell() + 1):
        print(length)
        raise Exception("Length too long.")

    # convert length from bytes to 16-bit Unicode words
    length = int(length / 2)

    data = []
    for i in range(0, length):
        (c,) = struct.unpack("<H", fid.read(2))
        data.append(c)


# Define read_header function
def read_header(fid):
    """Reads the Intan File Format header from the given file."""

    # Check 'magic number' at beginning of file to make sure this is an Intan
    # Technologies RHD2000 data file.
    (magic_number,) = struct.unpack("<I", fid.read(4))
    if magic_number != int("c6912702", 16):
        raise Exception("Unrecognized file type.")

    header = {}
    # Read version number.
    version = {}
    (version["major"], version["minor"]) = struct.unpack("<hh", fid.read(4))
    header["version"] = version

    freq = {}

    # Read information of sampling rate and amplifier frequency settings.
    (header["sample_rate"],) = struct.unpack("<f", fid.read(4))
    (
        freq["dsp_enabled"],
        freq["actual_dsp_cutoff_frequency"],
        freq["actual_lower_bandwidth"],
        freq["actual_upper_bandwidth"],
        freq["desired_dsp_cutoff_frequency"],
        freq["desired_lower_bandwidth"],
        freq["desired_upper_bandwidth"],
    ) = struct.unpack("<hffffff", fid.read(26))

    # This tells us if a software 50/60 Hz notch filter was enabled during
    # the data acquisition.
    (notch_filter_mode,) = struct.unpack("<h", fid.read(2))
    header["notch_filter_frequency"] = 0
    if notch_filter_mode == 1:
        header["notch_filter_frequency"] = 50
    elif notch_filter_mode == 2:
        header["notch_filter_frequency"] = 60
    freq["notch_filter_frequency"] = header["notch_filter_frequency"]

    (
        freq["desired_impedance_test_frequency"],
        freq["actual_impedance_test_frequency"],
    ) = struct.unpack("<ff", fid.read(8))

    note1 = read_qstring(fid)
    note2 = read_qstring(fid)
    note3 = read_qstring(fid)
    header["notes"] = {"note1": note1, "note2": note2, "note3": note3}

    # If data file is from GUI v1.1 or later, see if temperature sensor data was saved.
    header["num_temp_sensor_channels"] = 0
    if (version["major"] == 1 and version["minor"] >= 1) or (version["major"] > 1):
        (header["num_temp_sensor_channels"],) = struct.unpack("<h", fid.read(2))

    # If data file is from GUI v1.3 or later, load eval board mode.
    header["eval_board_mode"] = 0
    if ((version["major"] == 1) and (version["minor"] >= 3)) or (version["major"] > 1):
        (header["eval_board_mode"],) = struct.unpack("<h", fid.read(2))

    header["num_samples_per_data_block"] = 60
    # If data file is from v2.0 or later (Intan Recording Controller), load name of digital reference channel
    if version["major"] > 1:
        header["reference_channel"] = read_qstring(fid)
        header["num_samples_per_data_block"] = 128

    # Place frequency-related information in data structure. (Note: much of this structure is set above)
    freq["amplifier_sample_rate"] = header["sample_rate"]
    freq["aux_input_sample_rate"] = header["sample_rate"] / 4
    freq["supply_voltage_sample_rate"] = (
        header["sample_rate"] / header["num_samples_per_data_block"]
    )
    freq["board_adc_sample_rate"] = header["sample_rate"]
    freq["board_dig_in_sample_rate"] = header["sample_rate"]

    header["frequency_parameters"] = freq

    # Create structure arrays for each type of data channel.
    header["spike_triggers"] = []
    header["amplifier_channels"] = []
    header["aux_input_channels"] = []
    header["supply_voltage_channels"] = []
    header["board_adc_channels"] = []
    header["board_dig_in_channels"] = []
    header["board_dig_out_channels"] = []

    # Read signal summary from data file header.

    (number_of_signal_groups,) = struct.unpack("<h", fid.read(2))

    for signal_group in range(1, number_of_signal_groups + 1):
        signal_group_name = read_qstring(fid)
        signal_group_prefix = read_qstring(fid)
        (
            signal_group_enabled,
            signal_group_num_channels,
            signal_group_num_amp_channels,
        ) = struct.unpack("<hhh", fid.read(6))

        if (signal_group_num_channels > 0) and (signal_group_enabled > 0):
            for signal_channel in range(0, signal_group_num_channels):
                new_channel = {
                    "port_name": signal_group_name,
                    "port_prefix": signal_group_prefix,
                    "port_number": signal_group,
                }
                new_channel["native_channel_name"] = read_qstring(fid)
                new_channel["custom_channel_name"] = read_qstring(fid)
                (
                    new_channel["native_order"],
                    new_channel["custom_order"],
                    signal_type,
                    channel_enabled,
                    new_channel["chip_channel"],
                    new_channel["board_stream"],
                ) = struct.unpack("<hhhhhh", fid.read(12))
                new_trigger_channel = {}
                (
                    new_trigger_channel["voltage_trigger_mode"],
                    new_trigger_channel["voltage_threshold"],
                    new_trigger_channel["digital_trigger_channel"],
                    new_trigger_channel["digital_edge_polarity"],
                ) = struct.unpack("<hhhh", fid.read(8))
                (
                    new_channel["electrode_impedance_magnitude"],
                    new_channel["electrode_impedance_phase"],
                ) = struct.unpack("<ff", fid.read(8))

                if channel_enabled:
                    if signal_type == 0:
                        header["amplifier_channels"].append(new_channel)
                        header["spike_triggers"].append(new_trigger_channel)
                    elif signal_type == 1:
                        header["aux_input_channels"].append(new_channel)
                    elif signal_type == 2:
                        header["supply_voltage_channels"].append(new_channel)
                    elif signal_type == 3:
                        header["board_adc_channels"].append(new_channel)
                    elif signal_type == 4:
                        header["board_dig_in_channels"].append(new_channel)
                    elif signal_type == 5:
                        header["board_dig_out_channels"].append(new_channel)
                    else:
                        raise Exception("Unknown channel type.")

    # Summarize contents of data file.
    header["num_amplifier_channels"] = len(header["amplifier_channels"])
    header["num_aux_input_channels"] = len(header["aux_input_channels"])
    header["num_supply_voltage_channels"] = len(header["supply_voltage_channels"])
    header["num_board_adc_channels"] = len(header["board_adc_channels"])
    header["num_board_dig_in_channels"] = len(header["board_dig_in_channels"])
    header["num_board_dig_out_channels"] = len(header["board_dig_out_channels"])

    return header


# Define read_one_data_block function
def read_one_data_block(data, header, indices, fid):
    """Reads one 60 or 128 sample data block from fid into data, at the location indicated by indices."""

    # In version 1.2, we moved from saving timestamps as unsigned
    # integers to signed integers to accommodate negative (adjusted)
    # timestamps for pretrigger data['
    if (header["version"]["major"] == 1 and header["version"]["minor"] >= 2) or (
        header["version"]["major"] > 1
    ):
        data["t_amplifier"][
            indices["amplifier"] : (
                indices["amplifier"] + header["num_samples_per_data_block"]
            )
        ] = np.array(
            struct.unpack(
                "<" + "i" * header["num_samples_per_data_block"],
                fid.read(4 * header["num_samples_per_data_block"]),
            )
        )
    else:
        data["t_amplifier"][
            indices["amplifier"] : (
                indices["amplifier"] + header["num_samples_per_data_block"]
            )
        ] = np.array(
            struct.unpack(
                "<" + "I" * header["num_samples_per_data_block"],
                fid.read(4 * header["num_samples_per_data_block"]),
            )
        )

    if header["num_amplifier_channels"] > 0:
        tmp = np.fromfile(
            fid,
            dtype="uint16",
            count=header["num_samples_per_data_block"]
            * header["num_amplifier_channels"],
        )
        data["amplifier_data"][
            range(header["num_amplifier_channels"]),
            (indices["amplifier"]) : (
                indices["amplifier"] + header["num_samples_per_data_block"]
            ),
        ] = tmp.reshape(
            header["num_amplifier_channels"], header["num_samples_per_data_block"]
        )

    if header["num_aux_input_channels"] > 0:
        tmp = np.fromfile(
            fid,
            dtype="uint16",
            count=int(
                (header["num_samples_per_data_block"] / 4)
                * header["num_aux_input_channels"]
            ),
        )
        data["aux_input_data"][
            range(header["num_aux_input_channels"]),
            indices["aux_input"] : int(
                indices["aux_input"] + (header["num_samples_per_data_block"] / 4)
            ),
        ] = tmp.reshape(
            header["num_aux_input_channels"],
            int(header["num_samples_per_data_block"] / 4),
        )

    if header["num_supply_voltage_channels"] > 0:
        tmp = np.fromfile(
            fid, dtype="uint16", count=1 * header["num_supply_voltage_channels"]
        )
        data["supply_voltage_data"][
            range(header["num_supply_voltage_channels"]),
            indices["supply_voltage"] : (indices["supply_voltage"] + 1),
        ] = tmp.reshape(header["num_supply_voltage_channels"], 1)

    if header["num_temp_sensor_channels"] > 0:
        tmp = np.fromfile(
            fid, dtype="uint16", count=1 * header["num_temp_sensor_channels"]
        )
        data["temp_sensor_data"][
            range(header["num_temp_sensor_channels"]),
            indices["supply_voltage"] : (indices["supply_voltage"] + 1),
        ] = tmp.reshape(header["num_temp_sensor_channels"], 1)

    if header["num_board_adc_channels"] > 0:
        tmp = np.fromfile(
            fid,
            dtype="uint16",
            count=(header["num_samples_per_data_block"])
            * header["num_board_adc_channels"],
        )
        data["board_adc_data"][
            range(header["num_board_adc_channels"]),
            indices["board_adc"] : (
                indices["board_adc"] + header["num_samples_per_data_block"]
            ),
        ] = tmp.reshape(
            header["num_board_adc_channels"], header["num_samples_per_data_block"]
        )

    if header["num_board_dig_in_channels"] > 0:
        data["board_dig_in_raw"][
            indices["board_dig_in"] : (
                indices["board_dig_in"] + header["num_samples_per_data_block"]
            )
        ] = np.array(
            struct.unpack(
                "<" + "H" * header["num_samples_per_data_block"],
                fid.read(2 * header["num_samples_per_data_block"]),
            )
        )

    if header["num_board_dig_out_channels"] > 0:
        data["board_dig_out_raw"][
            indices["board_dig_out"] : (
                indices["board_dig_out"] + header["num_samples_per_data_block"]
            )
        ] = np.array(
            struct.unpack(
                "<" + "H" * header["num_samples_per_data_block"],
                fid.read(2 * header["num_samples_per_data_block"]),
            )
        )


# Define data_to_result function
def data_to_result(header, data, data_present):
    """Moves the header and data (if present) into a common object."""

    result = {}
    if header["num_amplifier_channels"] > 0 and data_present:
        result["t_amplifier"] = data["t_amplifier"]
    if header["num_aux_input_channels"] > 0 and data_present:
        result["t_aux_input"] = data["t_aux_input"]
    if header["num_supply_voltage_channels"] > 0 and data_present:
        result["t_supply_voltage"] = data["t_supply_voltage"]
    if header["num_board_adc_channels"] > 0 and data_present:
        result["t_board_adc"] = data["t_board_adc"]
    if (
        header["num_board_dig_in_channels"] > 0
        or header["num_board_dig_out_channels"] > 0
    ) and data_present:
        result["t_dig"] = data["t_dig"]
    if header["num_temp_sensor_channels"] > 0 and data_present:
        result["t_temp_sensor"] = data["t_temp_sensor"]

    if header["num_amplifier_channels"] > 0:
        result["spike_triggers"] = header["spike_triggers"]

    result["notes"] = header["notes"]
    result["frequency_parameters"] = header["frequency_parameters"]

    if header["version"]["major"] > 1:
        result["reference_channel"] = header["reference_channel"]

    if header["num_amplifier_channels"] > 0:
        result["amplifier_channels"] = header["amplifier_channels"]
        if data_present:
            result["amplifier_data"] = data["amplifier_data"]

    if header["num_aux_input_channels"] > 0:
        result["aux_input_channels"] = header["aux_input_channels"]
        if data_present:
            result["aux_input_data"] = data["aux_input_data"]

    if header["num_supply_voltage_channels"] > 0:
        result["supply_voltage_channels"] = header["supply_voltage_channels"]
        if data_present:
            result["supply_voltage_data"] = data["supply_voltage_data"]

    if header["num_board_adc_channels"] > 0:
        result["board_adc_channels"] = header["board_adc_channels"]
        if data_present:
            result["board_adc_data"] = data["board_adc_data"]

    if header["num_board_dig_in_channels"] > 0:
        result["board_dig_in_channels"] = header["board_dig_in_channels"]
        if data_present:
            result["board_dig_in_data"] = data["board_dig_in_data"]

    if header["num_board_dig_out_channels"] > 0:
        result["board_dig_out_channels"] = header["board_dig_out_channels"]
        if data_present:
            result["board_dig_out_data"] = data["board_dig_out_data"]

    return result


# Define load_file function
def load_file(filename):
    # Open file
    fid = open(filename, "rb")
    filesize = os.path.getsize(filename)

    # Read file header
    header = read_header(fid)

    # Determine how many samples the data file contains
    bytes_per_block = get_bytes_per_data_block(header)

    # Calculate how many data blocks are present
    data_present = False
    bytes_remaining = filesize - fid.tell()
    if bytes_remaining > 0:
        data_present = True

    if bytes_remaining % bytes_per_block != 0:
        raise Exception(
            "Something is wrong with file size : should have a whole number of data blocks"
        )

    num_data_blocks = int(bytes_remaining / bytes_per_block)

    # Calculate how many samples of each signal type are present
    num_amplifier_samples = header["num_samples_per_data_block"] * num_data_blocks
    num_aux_input_samples = int(
        (header["num_samples_per_data_block"] / 4) * num_data_blocks
    )
    num_supply_voltage_samples = 1 * num_data_blocks
    num_board_adc_samples = header["num_samples_per_data_block"] * num_data_blocks
    num_board_dig_in_samples = header["num_samples_per_data_block"] * num_data_blocks
    num_board_dig_out_samples = header["num_samples_per_data_block"] * num_data_blocks

    if data_present:
        # Pre-allocate memory for data
        data = {}
        if (header["version"]["major"] == 1 and header["version"]["minor"] >= 2) or (
            header["version"]["major"] > 1
        ):
            data["t_amplifier"] = np.zeros(num_amplifier_samples, dtype=np.int_)
        else:
            data["t_amplifier"] = np.zeros(num_amplifier_samples, dtype=np.uint)

        data["amplifier_data"] = np.zeros(
            [header["num_amplifier_channels"], num_amplifier_samples], dtype=np.uint
        )
        data["aux_input_data"] = np.zeros(
            [header["num_aux_input_channels"], num_aux_input_samples], dtype=np.uint
        )
        data["supply_voltage_data"] = np.zeros(
            [header["num_supply_voltage_channels"], num_supply_voltage_samples],
            dtype=np.uint,
        )
        data["temp_sensor_data"] = np.zeros(
            [header["num_temp_sensor_channels"], num_supply_voltage_samples],
            dtype=np.uint,
        )
        data["board_adc_data"] = np.zeros(
            [header["num_board_adc_channels"], num_board_adc_samples], dtype=np.uint
        )

        data["board_dig_in_data"] = np.zeros(
            [header["num_board_dig_in_channels"], num_board_dig_in_samples],
            dtype=np.bool_,
        )
        data["board_dig_in_raw"] = np.zeros(num_board_dig_in_samples, dtype=np.uint)

        data["board_dig_out_data"] = np.zeros(
            [header["num_board_dig_out_channels"], num_board_dig_out_samples],
            dtype=np.bool_,
        )
        data["board_dig_out_raw"] = np.zeros(num_board_dig_out_samples, dtype=np.uint)

        # Initialize indices used in looping
        indices = {}
        indices["amplifier"] = 0
        indices["aux_input"] = 0
        indices["supply_voltage"] = 0
        indices["board_adc"] = 0
        indices["board_dig_in"] = 0
        indices["board_dig_out"] = 0

        for i in range(num_data_blocks):
            read_one_data_block(data, header, indices, fid)

            # Increment indices
            indices["amplifier"] += header["num_samples_per_data_block"]
            indices["aux_input"] += int(header["num_samples_per_data_block"] / 4)
            indices["supply_voltage"] += 1
            indices["board_adc"] += header["num_samples_per_data_block"]
            indices["board_dig_in"] += header["num_samples_per_data_block"]
            indices["board_dig_out"] += header["num_samples_per_data_block"]

        # Make sure we have read exactly the right amount of data
        bytes_remaining = filesize - fid.tell()
        if bytes_remaining != 0:
            raise Exception("Error: End of file not reached.")

    fid.close()

    if data_present:
        # Extract digital input channels to separate variables
        for i in range(header["num_board_dig_in_channels"]):
            data["board_dig_in_data"][i, :] = np.not_equal(
                np.bitwise_and(
                    data["board_dig_in_raw"],
                    (1 << header["board_dig_in_channels"][i]["native_order"]),
                ),
                0,
            )

        # Extract digital output channels to separate variables
        for i in range(header["num_board_dig_out_channels"]):
            data["board_dig_out_data"][i, :] = np.not_equal(
                np.bitwise_and(
                    data["board_dig_out_raw"],
                    (1 << header["board_dig_out_channels"][i]["native_order"]),
                ),
                0,
            )

        # Scale voltage levels appropriately
        data["amplifier_data"] = np.multiply(
            0.195, (data["amplifier_data"].astype(np.int32) - 32768)
        )  # units = microvolts
        data["aux_input_data"] = np.multiply(
            37.4e-6, data["aux_input_data"]
        )  # units = volts
        data["supply_voltage_data"] = np.multiply(
            74.8e-6, data["supply_voltage_data"]
        )  # units = volts
        if header["eval_board_mode"] == 1:
            data["board_adc_data"] = np.multiply(
                152.59e-6, (data["board_adc_data"].astype(np.int32) - 32768)
            )  # units = volts
        elif header["eval_board_mode"] == 13:
            data["board_adc_data"] = np.multiply(
                312.5e-6, (data["board_adc_data"].astype(np.int32) - 32768)
            )  # units = volts
        else:
            data["board_adc_data"] = np.multiply(
                50.354e-6, data["board_adc_data"]
            )  # units = volts
        data["temp_sensor_data"] = np.multiply(
            0.01, data["temp_sensor_data"]
        )  # units = deg C

        # Check for gaps in timestamps
        num_gaps = np.sum(
            np.not_equal(data["t_amplifier"][1:] - data["t_amplifier"][:-1], 1)
        )

        # Scale time steps (units = seconds)
        data["t_amplifier"] = data["t_amplifier"] / header["sample_rate"]
        data["t_aux_input"] = data["t_amplifier"][range(0, len(data["t_amplifier"]), 4)]
        data["t_supply_voltage"] = data["t_amplifier"][
            range(0, len(data["t_amplifier"]), header["num_samples_per_data_block"])
        ]
        data["t_board_adc"] = data["t_amplifier"]
        data["t_dig"] = data["t_amplifier"]
        data["t_temp_sensor"] = data["t_supply_voltage"]

    else:
        data = []

    # Move variables to result struct
    result = data_to_result(header, data, data_present)

    return result

if __name__ == "__main__":
    from pathlib import Path

    data_path = Path("/Users/vigji/Downloads/exampleephysdata")
    intan_path = data_path / "IntanData"
    t = IntanData(intan_path).time_array
    print(np.histogram(np.diff(t)))