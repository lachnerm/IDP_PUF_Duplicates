import h5py

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms


class CenterCrop(torch.nn.Module):

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, img):
        return center_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size=512)'


def center_crop(img, crop_size):
    """
    Crops the center part of given size of an image (2d numpy array).

    :param img: image to be cropped (2d numpy array)
    :param crop_size: size of the center square that will be cropped from the image
    :return: cropped center part of the image
    """
    y_size, x_size = img.shape[-2:]
    x_start = x_size // 2 - (crop_size // 2)
    y_start = y_size // 2 - (crop_size // 2)
    if len(img.shape) == 2:
        return img[y_start:y_start + crop_size, x_start:x_start + crop_size]
    else:
        return img[:, y_start:y_start + crop_size, x_start:x_start + crop_size]


class Dataset(Dataset):
    def __init__(self, data_path, ids, transform=None):
        """
        Dataset wrapper for the DL model. Directly accesses a hdf5 dataset, transforms the data (if any
        transformations are defined), and returns the samples to a data loader.

        :param data_path: name of the folder where the data file is stored
        :param ids: ids of the samples that will be accessed via the dataset
        :param transform: sequence of transformations that will be applied to the response before returning a CRP
        """
        self.data_path = data_path
        self._h5_gen = None

        self.normalize = transforms.Normalize((0.5,), (0.5,))
        self.denormalize = transforms.Normalize(mean=[-1.0], std=[2.0])

        self.folder = data_path
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if self._h5_gen is None:
            self._h5_gen = self._get_generator(self.data_path)
            next(self._h5_gen)

        challenge, response = self._h5_gen.send(self.ids[idx])
        challenge = torch.tensor(challenge, dtype=torch.float)

        if self.transform:
            response = self.transform(response)
        response = self.normalize(response)

        return challenge, response

    def _get_generator(self, path):
        with h5py.File(path, 'r') as data:
            index = yield
            while True:
                c = data.get("challenges")[index]
                #r = data.get("responses")[index]
                r = data.get("responses_1")[index]
                index = yield c, r


class DataModule(LightningDataModule):

    def __init__(self, batch_size, folder, training_ids, val_ids, test_ids):
        """
        Initializes the data module used for the DL model. The data is split into a training, validation and test set.
        Provides three dataloaders to access all datasets.

        :param batch_size: batch size for the returned batches
        :param folder: name of the folder where the data file is stored
        :param training_ids: ids of the training samples
        :param val_ids: ids of the validation samples
        :param test_ids: ids of the test samples
        """
        super().__init__()
        self.batch_size = batch_size
        self.folder = folder
        self.train_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": 4,
            "pin_memory": True,
            "shuffle": True
        }
        self.val_test_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": 4,
            "pin_memory": True
        }
        self.training_ids = training_ids
        self.val_ids = val_ids
        self.test_ids = test_ids

    def setup(self):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.train_dataset = Dataset(self.folder, self.training_ids, transform)
        self.val_dataset = Dataset(self.folder, self.val_ids, transform)
        self.test_dataset = Dataset(self.folder, self.test_ids, transform)

        self.denormalize = self.train_dataset.denormalize

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.train_kwargs)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, **self.val_test_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.val_test_kwargs)
