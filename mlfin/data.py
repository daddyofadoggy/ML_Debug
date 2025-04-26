import kagglehub

import pandas as pd

from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset


class MiamiDataset(Dataset):
    """The miami housing dataset which we will pull from kaggle."""

    def __init__(self):
        """Dataset initialization, may involve downloading the dataset."""

        path = kagglehub.dataset_download("deepcontractor/miami-housing-dataset")
        df = pd.read_csv(path + "/miami-housing.csv")
        self.df = df

        self.features = torch.tensor(
            df.drop(columns=["SALE_PRC", "PARCELNO"]).values
        ).to(dtype=torch.float32)
        self.scaler = StandardScaler()
        self.features = torch.tensor(
            self.scaler.fit_transform(self.features.cpu().numpy())
        ).to(dtype=torch.float32)

        self.n_features = self.features.shape[1]
        self.targets = (
            torch.tensor(df["SALE_PRC"].values)
            .to(dtype=torch.float32)
            .unsqueeze(1)
        )
        self.targets = torch.tensor(
            StandardScaler().fit_transform(self.targets.cpu().numpy())
        ).to(dtype=torch.float32)

        self.n_targets = self.targets.shape[1]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
