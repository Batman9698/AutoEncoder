from torch.utils.data import Dataset
import pandas as pd
import numpy as np
class MNIST(Dataset):
    # ToTensor() transform should be included else it'll cause an error
    def __init__(self, file_path, transform = None):
        self.data = pd.read_csv(file_path)
        self.transform = transform
        
    def __len__(self):
        return(len(self.data))
    
    def __getitem__(self, index):
        image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((28, 28,1))
        label = self.data.iloc[index, 0]
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label