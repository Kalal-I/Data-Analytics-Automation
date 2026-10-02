import os

import pandas as pd


def read_csv_files():
    """Prompt for a path and read one or more CSV files."""
    path = input("Enter the path to the CSV file or directory: ")

    if os.path.isfile(path):
        if os.path.splitext(path)[1].lower() != ".csv":
            raise ValueError("Expected a CSV file, got: {}".format(path))
        csv_paths = [path]
    elif os.path.isdir(path):
        csv_paths = sorted(
            os.path.join(path, filename)
            for filename in os.listdir(path)
            if os.path.isfile(os.path.join(path, filename))
            and os.path.splitext(filename)[1].lower() == ".csv"
        )
        if not csv_paths:
            raise FileNotFoundError(
                "No CSV files found in directory: {}".format(path)
            )
    else:
        raise FileNotFoundError("Path does not exist: {}".format(path))

    dataframes = [pd.read_csv(csv_path) for csv_path in csv_paths]
    return dataframes[0] if len(dataframes) == 1 else dataframes