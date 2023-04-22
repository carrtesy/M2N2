import copy

import torch
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import pickle
from utils.metrics import get_summary_stats
from sklearn.metrics import roc_curve

matplotlib.rcParams['agg.path.chunksize'] = 10000


class Tester:
    '''
    Test-time logics,
    including offline evaluation and online adaptation.
    '''
    def __init__(self, args, logger, train_loader, test_loader):
        self.args = args
        self.logger = logger
        self.train_loader = train_loader
        self.test_loader = test_loader


    def calculate_recon_errors(self, dataloader):
        pass


    def checkpoint(self, filepath):
        self.logger.info(f"checkpointing: {filepath} @Trainer - torch.save")
        torch.save(self.model.state_dict(), filepath)


    def load(self, filepath):
        self.logger.info(f"loading: {filepath} @Trainer - torch.load_state_dict")
        self.model.load_state_dict(torch.load(filepath))
        self.model.to(self.args.device)


    @staticmethod
    def save_dictionary(dictionary, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(dictionary, f)