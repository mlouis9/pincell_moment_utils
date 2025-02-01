import numpy as np
import h5py

# Input CSV file
csv_file = "cas14.csv"  # Replace with the actual CSV file name
h5_file = "cas14.h5"  # Output HDF5 file name

# Read the CSV file and parse the data
data = np.genfromtxt(csv_file, delimiter=",", dtype=float)

# Extract indices and values
indices = data[:, 0].astype(int)  # Left column as integers (indices)
values = data[:, 1]               # Right column as values

# Create an array with zeros of the appropriate size
max_index = np.max(indices)  # Determine the maximum index
array = np.zeros(max_index + 1)  # Create an array large enough to hold all indices

# Fill the array with the values at the specified indices
array[indices] = values

# Create an HDF5 file and store the array in a dataset
with h5py.File(h5_file, "w") as hdf:
    hdf.create_dataset("energy groups", data=array)

print(f"HDF5 file '{h5_file}' created with dataset 'energy groups'.")