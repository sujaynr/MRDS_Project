import h5py

def load_h5_file(file_path):
    """
    Load and read data from an HDF5 (.h5) file.

    Parameters:
    file_path (str): Path to the .h5 file.

    Returns:
    dict: A dictionary containing datasets and groups from the HDF5 file.
    """
    data = {}
    try:
        with h5py.File(file_path, 'r') as h5_file:
            def recursively_load_h5(file_obj, data_dict):
                for key in file_obj:
                    if isinstance(file_obj[key], h5py.Group):
                        # Recursively read groups
                        data_dict[key] = {}
                        recursively_load_h5(file_obj[key], data_dict[key])
                    elif isinstance(file_obj[key], h5py.Dataset):
                        # Load dataset into memory
                        data_dict[key] = file_obj[key][:]
                        # Print the shape of the dataset
                        print(f"Dataset: {key}, Shape: {file_obj[key].shape}")
            recursively_load_h5(h5_file, data)
        print(f"Successfully loaded file: {file_path}")
    except Exception as e:
        print(f"Failed to load the file {file_path}: {e}")
    return data

if __name__ == "__main__":
    # File paths
    fault_file_path = 'prepared_data_TILES/faultData.h5'
    geoAge_file_path = 'prepared_data_TILES/geoAge.h5'
    elevation_file_path = '/home/sujaynair/MRDS_Project/all_elevations.h5'

    # Example usage for loading and printing data shapes
    for file_path in [fault_file_path, geoAge_file_path, elevation_file_path]:
        print(f"\nProcessing file: {file_path}")
        h5_data = load_h5_file(file_path)
        if h5_data:
            for dataset in h5_data:
                if isinstance(h5_data[dataset], dict):
                    print(f"Group: {dataset}")
                else:
                    print(f"Dataset: {dataset}, Data Type: {type(h5_data[dataset])}")
