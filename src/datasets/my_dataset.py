from torch.utils.data import Dataset
import struct
from array import array
import numpy as np

class MyDataset(Dataset):
    """
    Standard PyTorch Dataset for individual data loading.
    """
    def __init__(self, data_path, label_path):
        super().__init__()

        imgs, labels = self.read_images_labels(data_path, label_path)
        self.samples = list(zip(imgs, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        one_hot_label = np.zeros(10)
        one_hot_label[label] = 1.0
        
        return img, one_hot_label
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append(None)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i] = img
        
        return images, labels
    
if __name__ == "__main__":
    dataset = MyDataset(
        data_path='/home/kimds/ml_template/data/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte',
        label_path='/home/kimds/ml_template/data/mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
    )
    print(len(dataset))
    img, label = dataset[0]
    print(img.shape)
    print(label)