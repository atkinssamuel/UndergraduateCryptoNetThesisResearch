### Duplication Instructions
1. Clone repository
2. Ensure that your environment has tensorflow==1.14.0
3. Create the following directories:
    - weights/
    - weights/conv
    - weights/dense
    - data_management/np_dataset
4. Use the data_management folder to import the MNIST data into .npy files
5. Modify the main.py code to run the training and testing functions for the fully-connected and convolutional networks

### Miscellaneous
Convolutional layer guide: https://stackoverflow.com/questions/34619177/what-does-tf-nn-conv2d-do-in-tensorflow  
The MNIST dataset is publically avaialble here: http://yann.lecun.com/exdb/mnist/