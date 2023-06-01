import torch

class MRIDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transformation=None):
        self.labels = labels
        self.images = images
        self.transformation = transformation

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = torch.tensor(self.images[index])
        
        if self.transformation is not None:
            image = self.transformation(image)

        return {'tensor': image, 'label': self.labels[index]}
