from datetime import datetime
import os
from pathlib import Path


def get_latest_partition_data(data_dir: Path) -> str:
    """Load the latest date from the given directory

    Args:
        data_dir (Path): path for the data

    Returns:
        string: latest partition filename
    """
    latest_file = None
    latest_time = None

    file_list = os.listdir(data_dir)
    for file in file_list:
        try:
            mod_time = datetime.fromtimestamp(os.path.getmtime(os.path.join(data_dir, file)))
            if latest_time is None or mod_time > latest_time:
                latest_time = mod_time
                latest_file = file
        except Exception as e:
            print(f"Could not get modification time for {data_dir}: {e}")

    if latest_file is None:
        raise ValueError(f"No valid files found in directory {data_dir}")
    return latest_file
