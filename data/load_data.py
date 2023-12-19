import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import ast

from torch.utils.data import DataLoader, Dataset
from sklearn import preprocessing
import data.load_anoshift as la
import pickle
import gc

class DataFactory:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.logger.info(f"current location: {os.getcwd()}")

        self.dataset_fn_dict = {
            # Synthetic dataset
            "toyUSW": self.load_toyUSW,
            "NeurIPS-TS-UNI": self.load_NeurIPS_TS_UNI,
            "NeurIPS-TS-MUL": self.load_NeurIPS_TS_MUL,

            # Real Dataset
            "SWaT": self.load_SWaT,
            "WADI": self.load_WADI,
            "SMAP": self.load_SMAP,
            "MSL": self.load_MSL,
            "SMD": self.load_SMD,
            "yahoo":self.load_yahoo,
            "CreditCard": self.load_CreditCard,
        }

        self.transforms = {
            "minmax": preprocessing.MinMaxScaler(),
            "std": preprocessing.StandardScaler(),
        }


    def __call__(self):
        # prepare data
        self.logger.info(f"Preparing {self.args.dataset} ...")
        train_X, train_y, test_X, test_y = self.load()
        self.logger.info(
            f"train: X - {train_X.shape}, y - {train_y.shape} " +
            f"test: X - {test_X.shape}, y - {test_y.shape}"
        )
        assert np.isnan(train_X).sum() == 0 and np.isnan(train_y).sum() == 0
        assert np.isnan(test_X).sum() == 0 and np.isnan(test_y).sum() == 0
        self.logger.info(f"Complete.")

        # dataloader, dataset
        self.logger.info(f"Preparing dataloader...")
        train_dataset, train_loader, test_dataset, test_loader = self.prepare(
            train_X, train_y, test_X, test_y,
            window_size=self.args.window_size,
            stride=self.args.stride,
            eval_stride=self.args.eval_stride,
            batch_size=self.args.batch_size,
            eval_batch_size=self.args.eval_batch_size,
            train_shuffle=True,
            test_shuffle=False,
            scaler=self.args.scaler,
        )

        # shape in batch
        sample_X, sample_y = next(iter(train_loader))
        self.logger.info(f"total train dataset- {len(train_loader)}, "
                         f"batch_X - {sample_X.shape}, "
                         f"batch_y - {sample_y.shape}")

        sample_X, sample_y = next(iter(test_loader))
        self.logger.info(f"total test dataset- {len(test_loader)}, "
                         f"batch_X - {sample_X.shape}, "
                         f"batch_y - {sample_y.shape}")
        self.logger.info(f"Complete.")

        return train_dataset, train_loader, test_dataset, test_loader


    def load(self):
        if self.args.dataset not in self.dataset_fn_dict.keys():
            raise ValueError(f"{self.args.dataset} dataset does not exist.\nList of available datasets: {self.dataset_fn_dict.keys()}")
        else:
            if self.args.dataset in ["SMAP", "MSL", "SMD", "yahoo"]:
                if not hasattr(self.args, "dataset_id"):
                    raise ValueError(f"Argument --dataset_id should be given to {self.args.dataset} dataset.")
                return self.dataset_fn_dict[self.args.dataset](self.args.dataset_id)
            else:
                return self.dataset_fn_dict[self.args.dataset]()


    def prepare(self, train_X, train_y, test_X, test_y,
                window_size,
                stride,
                eval_stride,
                batch_size,
                eval_batch_size,
                train_shuffle,
                test_shuffle,
                scaler,
                ):

        transform = self.transforms[scaler]
        train_dataset = TSADStandardDataset(
            train_X, train_y,
            flag="train", transform=transform,
            window_size=window_size,
            stride=stride,
        )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=train_shuffle,
        )

        transform = train_dataset.transform
        test_dataset = TSADStandardDataset(
            test_X, test_y,
            flag="test", transform=transform,
            window_size=window_size,
            stride=eval_stride,
        )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=eval_batch_size,
            shuffle=test_shuffle,
        )

        return train_dataset, train_dataloader, test_dataset, test_dataloader


    @staticmethod
    def load_toyUSW():
        base_dir = "data/toyUSW"
        with open(os.path.join(base_dir, "train.npy"), 'rb') as f:
            train_X = np.load(f)
        with open(os.path.join(base_dir, "train_label.npy"), 'rb') as f:
            train_y = np.load(f)

        with open(os.path.join(base_dir, "test.npy"), 'rb') as f:
            test_X = np.load(f)
        with open(os.path.join(base_dir, "test_label.npy"), 'rb') as f:
            test_y = np.load(f)

        train_X, test_X = train_X.astype(np.float32), test_X.astype(np.float32)
        train_y, test_y = train_y.astype(int), test_y.astype(int)

        assert np.isnan(train_X).sum() == 0 and np.isnan(train_y).sum() == 0
        assert np.isnan(test_X).sum() == 0 and np.isnan(test_y).sum() == 0

        return train_X, train_y, test_X, test_y


    @staticmethod
    def load_NeurIPS_TS_UNI():
        base_dir = "data/NeurIPS-TS"
        normal = pd.read_csv(os.path.join(base_dir, "nts_uni_normal.csv"))
        abnormal = pd.read_csv(os.path.join(base_dir, "nts_uni_abnormal.csv"))

        train_X, train_y = normal.values[:, :-1], normal.values[:, -1]
        test_X, test_y = abnormal.values[:, :-1], abnormal.values[:, -1]

        train_X, test_X = train_X.astype(np.float32), test_X.astype(np.float32)
        train_y, test_y = train_y.astype(int), test_y.astype(int)

        assert np.isnan(train_X).sum() == 0 and np.isnan(train_y).sum() == 0
        assert np.isnan(test_X).sum() == 0 and np.isnan(test_y).sum() == 0

        return train_X, train_y, test_X, test_y


    @staticmethod
    def load_NeurIPS_TS_MUL():

        base_dir = "data/NeurIPS-TS"
        normal = pd.read_csv(os.path.join(base_dir, "nts_mul_normal.csv"))
        abnormal = pd.read_csv(os.path.join(base_dir, "nts_mul_abnormal.csv"))

        train_X, train_y = normal.values[:, :-1], normal.values[:, -1]
        test_X, test_y = abnormal.values[:, :-1], abnormal.values[:, -1]

        train_X, test_X = train_X.astype(np.float32), test_X.astype(np.float32)
        train_y, test_y = train_y.astype(int), test_y.astype(int)

        assert np.isnan(train_X).sum() == 0 and np.isnan(train_y).sum() == 0
        assert np.isnan(test_X).sum() == 0 and np.isnan(test_y).sum() == 0

        return train_X, train_y, test_X, test_y


    @staticmethod
    def load_SWaT():
        base_dir = "data/SWaT"
        SWAT_TRAIN_PATH = os.path.join(base_dir, 'SWaT_Dataset_Normal_v0.csv')
        SWAT_TEST_PATH = os.path.join(base_dir, 'SWaT_Dataset_Attack_v0.csv')
        df_train = pd.read_csv(SWAT_TRAIN_PATH, index_col=0)
        df_test = pd.read_csv(SWAT_TEST_PATH, index_col=0)

        def process_df(df):
            for var_index in [item for item in df.columns if item != 'Normal/Attack']:
                df[var_index] = pd.to_numeric(df[var_index], errors='coerce')
            df.reset_index(drop=True, inplace=True)
            df.fillna(method='ffill', inplace=True)

            X = df.values[:, :-1].astype(np.float32)
            y = (df['Normal/Attack'] == 'Attack').to_numpy().astype(int)
            return X, y

        train_X, train_y = process_df(df_train)
        test_X, test_y = process_df(df_test)

        train_X, test_X = train_X.astype(np.float32), test_X.astype(np.float32)
        train_y, test_y = train_y.astype(int), test_y.astype(int)

        assert np.isnan(train_X).sum() == 0 and np.isnan(train_y).sum() == 0
        assert np.isnan(test_X).sum() == 0 and np.isnan(test_y).sum() == 0

        return train_X, train_y, test_X, test_y


    @staticmethod
    def load_WADI():
        base_dir = "data/WADI"
        WADI_TRAIN_PATH = os.path.join(base_dir, 'WADI_14days_new.csv')
        WADI_TEST_PATH = os.path.join(base_dir, 'WADI_attackdataLABLE.csv')
        df_train = pd.read_csv(WADI_TRAIN_PATH, index_col=0) # Drop row that has all NaN values
        df_train = df_train.dropna(how="all")
        df_test = pd.read_csv(WADI_TEST_PATH, index_col=0, skiprows=[0]) # apply skiprow or erase the first row
        df_train = df_train.dropna(thresh=2) # Keep only the rows with at least 2 non-NA values.

        # df_train
        for col in df_train.columns:
            if col.strip() not in ["Date", "Time"]:
                df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
        df_train.fillna(method='ffill', inplace=True)
        df_train.fillna(method='bfill', inplace=True)
        df_train.dropna(axis='columns', inplace=True) # drop null columns

        train_X = df_train.values[:, 2:].astype(np.float32) # col0: Date, col1: Time
        T, C = train_X.shape
        train_y = np.zeros((T,), dtype=int)

        # df_test
        for col in df_test.columns:
            if col.strip() not in ["Date", "Time"]:
                df_test[col] = pd.to_numeric(df_test[col], errors='coerce')
        df_test.fillna(method='ffill', inplace=True)
        df_test.fillna(method='bfill', inplace=True)
        df_test.dropna(axis='columns', inplace=True) # drop null columns

        test_X = df_test.values[:, 2:-1].astype(np.float32) # col0: Date, col1: Time, col[-1]: label
        test_y = (df_test.values[:, -1] == -1).astype(int)

        train_X, test_X = train_X.astype(np.float32), test_X.astype(np.float32)
        train_y, test_y = train_y.astype(int), test_y.astype(int)

        assert np.isnan(train_X).sum() == 0 and np.isnan(train_y).sum() == 0
        assert np.isnan(test_X).sum() == 0 and np.isnan(test_y).sum() == 0

        return train_X, train_y, test_X, test_y


    @staticmethod
    def load_CreditCard():
        data_df = pd.read_csv("data/CreditCard/creditcard.csv", index_col=0)

        N = len(data_df)
        X, y = data_df.values[:, 0:-1].astype(np.float32), data_df.values[:,-1].astype(int)

        train_X, test_X = X[:N//2], X[N//2:]
        T, C = train_X.shape
        train_y, test_y = y[:N//2], y[N//2:]

        assert np.isnan(train_X).sum() == 0 and np.isnan(train_y).sum() == 0
        assert np.isnan(test_X).sum() == 0 and np.isnan(test_y).sum() == 0

        return train_X, train_y, test_X, test_y


    @staticmethod
    def load_SMAP(chan_id):
        base_dir = "data/SMAP"
        label = pd.read_csv(os.path.join(base_dir, "labeled_anomalies.csv"))
        SMAP_label = label[label["spacecraft"] == "SMAP"].sort_values(by="chan_id")

        train_X = np.load(os.path.join(base_dir, "train", f"{chan_id}.npy"))
        T, C = train_X.shape
        train_y = np.zeros((T,), dtype=int)

        test_X = np.load(os.path.join(base_dir, "test", f"{chan_id}.npy"))
        T, C = test_X.shape
        anos = ast.literal_eval(SMAP_label[SMAP_label["chan_id"] == chan_id]["anomaly_sequences"].values[0])
        test_y = np.zeros((T,), dtype=int)
        for ano in anos:
            s, e = ano
            test_y[s:e] = 1

        train_X, test_X = train_X.astype(np.float32), test_X.astype(np.float32)
        train_y, test_y = train_y.astype(int), test_y.astype(int)

        assert np.isnan(train_X).sum() == 0 and np.isnan(train_y).sum() == 0
        assert np.isnan(test_X).sum() == 0 and np.isnan(test_y).sum() == 0

        return train_X, train_y, test_X, test_y

    @staticmethod
    def load_MSL(chan_id):
        base_dir = "data/MSL"
        label = pd.read_csv(os.path.join(base_dir, "labeled_anomalies.csv"))
        MSL_label = label[label["spacecraft"] == "MSL"].sort_values(by="chan_id")

        train_X = np.load(os.path.join(base_dir, "train", f"{chan_id}.npy"))
        T, C = train_X.shape
        train_y = np.zeros((T,), dtype=int)

        test_X = np.load(os.path.join(base_dir, "test", f"{chan_id}.npy"))
        T, C = test_X.shape
        anos = ast.literal_eval(MSL_label[MSL_label["chan_id"] == chan_id]["anomaly_sequences"].values[0])
        test_y = np.zeros((T,), dtype=int)
        for ano in anos:
            s, e = ano
            test_y[s:e] = 1

        train_X, test_X = train_X.astype(np.float32), test_X.astype(np.float32)
        train_y, test_y = train_y.astype(int), test_y.astype(int)

        assert np.isnan(train_X).sum() == 0 and np.isnan(train_y).sum() == 0
        assert np.isnan(test_X).sum() == 0 and np.isnan(test_y).sum() == 0

        return train_X, train_y, test_X, test_y

    @staticmethod
    def load_yahoo(fname):
        base_dir = "data/yahoo/A1Benchmark"

        data = pd.read_csv(os.path.join(base_dir, f"{fname}.csv"))
        X = data["value"].values[:, None]
        label = data["is_anomaly"].values

        train_X = X[:400]
        T, C = train_X.shape
        train_y = np.zeros((T,), dtype=int)

        test_X = X[400:]
        T, C = test_X.shape
        test_y = label[400:]

        train_X, test_X = train_X.astype(np.float32), test_X.astype(np.float32)
        train_y, test_y = train_y.astype(int), test_y.astype(int)

        assert np.isnan(train_X).sum() == 0 and np.isnan(train_y).sum() == 0
        assert np.isnan(test_X).sum() == 0 and np.isnan(test_y).sum() == 0

        return train_X, train_y, test_X, test_y


    @staticmethod
    def load_SMD(machine_id):
        base_dir = "data/SMD"
        train_df = pd.read_csv(os.path.join(base_dir, "train", f"{machine_id}.txt"), header=None)
        test_df = pd.read_csv(os.path.join(base_dir, "test", f"{machine_id}.txt"), header=None)
        test_label_df = pd.read_csv(os.path.join(base_dir, "test_label", f"{machine_id}.txt"), header=None)

        train_X = train_df.to_numpy()
        T, C = train_X.shape
        train_y = np.zeros((T,), dtype=int)
        test_X = test_df.to_numpy()
        test_y = test_label_df.to_numpy().squeeze(1)

        train_X, test_X = train_X.astype(np.float32), test_X.astype(np.float32)
        train_y, test_y = train_y.astype(int), test_y.astype(int)

        assert np.isnan(train_X).sum() == 0 and np.isnan(train_y).sum() == 0
        assert np.isnan(test_X).sum() == 0 and np.isnan(test_y).sum() == 0

        return train_X, train_y, test_X, test_y

    @staticmethod
    def visualize_dataset(train_X, train_y, test_X, test_y, dataset_name, feature_idx):
        '''
        visualize the trajectory of certain feature.
        :param dataset_name
        :param feature_idx
        '''

        F = feature_idx
        # plot normal
        plt.figure(figsize=(20, 6))
        plt.plot(train_X[:, F])
        plt.title(f"{dataset_name} train data, feature {F}")
        plt.show()

        # plot abnormal
        plt.figure(figsize=(20, 6))
        plt.plot(test_X[:, F])
        plt.title(f"{dataset_name} test data, feature {F}")
        s, e = None, None
        for i in range(len(test_X)):
            if test_y[i] == 1 and s is None:
                s = i
            elif test_y[i] == 0 and s is not None:
                e = i - 1
                if (e - s) > 0:
                    plt.axvspan(s, e, facecolor='red', alpha=0.5)
                else:
                    plt.plot(s, test_X[s, F], 'ro')
                s, e = None, None
        plt.show()


class TSADStandardDataset(Dataset):
    def __init__(self, X:np.array, y:np.array, flag, transform, window_size, stride):
        super().__init__()
        self.transform = transform
        if flag == "train":
            self.transform.fit(X)
        # raw X, raw y (before window fitting)
        self.rX = X
        self.ry = y

        # fit to window size
        self.W = window_size
        self.S = stride
        self.T = X.shape[0] # total timesteps
        d = (self.T - self.W) // self.S + 1  # number of length W chunks.
        self.len = d

        # timestep remainders: concat last W observation, label with -1.
        r = self.T - (self.W + (d-1) * self.S)
        if r != 0:
            self.len += 1
            n_repeat = self.W - r
            X = np.concatenate([X[:d*self.W], X[-self.W:]])
            y = np.concatenate([y[:d*self.W], np.array([-1]*n_repeat), y[d*self.W:]]) # 0 for normals, 1 for anomalies, -1 for paddings.

        self.X = self.transform.transform(X)
        self.y = y


    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        _idx = idx * self.S
        X, y = self.X[_idx:_idx+self.W], self.y[_idx:_idx+self.W]
        return X, y


if __name__ == "__main__":
    test_X, test_y = np.random.randn(449919, 51), np.zeros(449919)

    W, S = 12, 1

    data = TSADStandardDataset(
        test_X, test_y,
        flag="test", transform=preprocessing.StandardScaler(),
        window_size=W,
        stride=S,
    )