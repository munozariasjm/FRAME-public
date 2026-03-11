# pnb/data_loader.py

import os
import glob
import pandas as pd
from .config import PnBConfig

class MultiIsotopeDataLoader:
    """
    Loads multiple isotope CSV files from a directory, merges them, and
    extracts Z and N from the filename.
    """
    def __init__(self, config: PnBConfig):
        self.config = config
        self.files = glob.glob(os.path.join(self.config.data_directory, self.config.file_pattern))
        if not self.files:
            raise FileNotFoundError(
                f"No files found with pattern '{self.config.file_pattern}' in '{self.config.data_directory}'"
            )
        print(f"[DataLoader] Found {len(self.files)} files.")

    def load_all_data(self) -> pd.DataFrame:
        """Loads all CSVs and returns a combined DataFrame."""
        all_data = []
        for file_path in self.files:
            df = pd.read_csv(file_path)
            basename = os.path.basename(file_path)
            token = basename.split("_")[0]
            element = "".join(filter(str.isalpha, token))
            mass_str = "".join(filter(str.isdigit, token))
            if not mass_str:
                raise ValueError(f"No mass number found in filename: {basename}")
            mass_number = int(mass_str)

            Z = self.element_to_Z(element)
            if Z == 0: raise ValueError(f"Element '{element}' not recognized.")

            df['Z'] = Z
            df['N'] = mass_number - Z
            all_data.append(df)

        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"[DataLoader] Combined data shape: {combined_data.shape}")
        return combined_data

    def element_to_Z(self, element: str) -> int:
        """Converts element symbol to atomic number Z."""
        periodic_table = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Ca': 20, 'Al': 13}
        return periodic_table.get(element, 0)