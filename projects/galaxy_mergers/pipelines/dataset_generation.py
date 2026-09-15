"""
This script turns a directory structured like:

into a dataset suitable for training machine learning models.
The structure of the dataset is as follows:
preprocessed.hdf5
├── images
│   shape = (N, H, W, C)
│   dtype = float32
├── ids
│   shape = (N,)
│   dtype = int64

And a metadata file containing information about the dataset:
preprocessed.parquet
├── index
│   shape = (N,)
│   dtype = int64
├── object_id
│   shape = (N,)
│   dtype = int64
├── snapshot
│   shape = (N,)
│   dtype = int64
├── split
│   shape = (N,)
│   dtype = object
├── redshift
│   shape = (N,)
│   dtype = float32

Example of parquet metadata:
index | object_id | snapshot | split | redshift |
-------------------------------------------------
0     | 1000298   | 0064     | train | 3.00     | 
1     | 1000298   | 0066     | train | 2.75     |
2     | 2000134   | 0088     | val   | 1.25     |
...

Also produces 3 split datasets: train, val, and test.
They are saved as txt files and contain the indices of the samples belonging to each split.

This allows for easy access to the training, validation, and test splits without having to
parse the metadata file each time.

The structure of the directory produced by this script is as follows:
dataset/
├── preprocessed.hdf5
├── preprocessed.parquet
├── train.txt
├── val.txt
└── test.txt
"""