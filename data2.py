import os
import struct
import numpy as np

def extract_data(img_path, lbl_path):
    with open(lbl_path, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl_data = np.fromfile(flbl, dtype=np.int8)

    with open(img_path, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img_data = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl_data), rows, cols)
    return img_data, lbl_data

def normalize(X):
    return 2.*(X/255.-0.5)

def get_data(dataset, norm, one_hot):

    path = "./data/" + dataset
    file_path_train_img = "./data/mnist/train-images.idx3-ubyte"#os.path.join(path, 'train-images-idx3-ubyte')
    file_path_train_lbl = "./data/mnist/train-labels.idx1-ubyte"#os.path.join(path, 'train-labels-idx1-ubyte')
    file_path_test_img = "./data/mnist/t10k-images.idx3-ubyte"#os.path.join(path, 't10k-images-idx3-ubyte')
    file_path_test_lbl = "./data/mnist/t10k-labels.idx1-ubyte" # os.path.join(path, 't10k-labels-idx1-ubyte')

    # Extract data from files
    img_train, lbl_train = extract_data(file_path_train_img, file_path_train_lbl)
    img_test, lbl_test = extract_data(file_path_test_img, file_path_test_lbl)

    print(img_train.shape)
    # np.savetxt('5.csv', img_train[:100], delimiter=',')
    # Normalize image data
    if norm:
        img_train, img_test = normalize(img_train), normalize(img_test)

    # Reshape images to vectors
    n_input = img_train.shape[-1]**2
    img_train, img_test = img_train.reshape(len(img_train), n_input), img_test.reshape(len(img_test), n_input)


    # img_train, img_test = normalize(img_train), normalize(img_test)

    np.savetxt('5.csv', np.array(img_train[:100], dtype=int), delimiter=';')

    print(img_train.shape)
    # Compute one-hot labels
    n_classes = len(np.unique(lbl_train))
    if one_hot:
        lbl_train = np.eye(n_classes)[lbl_train]
        lbl_test = np.eye(n_classes)[lbl_test]

    # Split evaluation dataset from training dataset
    prop = 0.95 # 5% of training data will be used for evaluation
    split_idx = int(prop*len(img_train))
    img_eval, lbl_eval = img_train[split_idx:], lbl_train[split_idx:]
    img_train, lbl_train = img_train[:split_idx], lbl_train[:split_idx]

    return img_train, lbl_train, img_eval, lbl_eval, img_test, lbl_test, n_classes, n_input
