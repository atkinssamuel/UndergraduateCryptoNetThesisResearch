### Duplication Instructions
1. Clone repository
2. Ensure that your environment has tensorflow==1.14.0
3. Create the following directories:
    - weights/
    - weights/conv
    - weights/dense
    - data_management/np_dataset
    - data_management/dataset
4. Extract the compressed dataset files into the data_management/dataset folder
5. In the data_management/dataset folder, replace the "."s in each of the dataset files with "-"s. 
6. In main.py, call import_MNIST() to import the dataset and populate the .npy dataset files.
6. Modify the main.py code to run the training and testing functions for the fully-connected and convolutional networks

### Miscellaneous
Convolutional layer guide: https://stackoverflow.com/questions/34619177/what-does-tf-nn-conv2d-do-in-tensorflow  
The MNIST dataset is publically avaialble here: http://yann.lecun.com/exdb/mnist/