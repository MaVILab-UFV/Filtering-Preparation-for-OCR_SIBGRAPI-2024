from pathlib import PurePath

import numpy as np
import fsspec

from kedro.io import AbstractDataSet
from kedro.io.core import get_filepath_str, get_protocol_and_path

class NumpyZipDataSet(AbstractDataSet):
    def __init__(self, filepath: str):
        protocol, path = get_protocol_and_path(filepath)
        self._filepath = PurePath(filepath)
        self._protocol = protocol
        self._fs = fsspec.filesystem(self._protocol)

    def _load(self):
        load_path = get_filepath_str(self._filepath, self._protocol)
        return np.load(load_path)

    def _save(self, data):
        save_path = get_filepath_str(self._filepath, self._protocol)
        np.savez_compressed(save_path, **data)

    def _describe(self):
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._filepath, protocol=self._protocol)
