import torch
import matplotlib
import pickle

matplotlib.rcParams['agg.path.chunksize'] = 10000


class Trainer:
    '''
    Prepares Offline-trained models.
    '''
    def __init__(self, args, logger, train_loader):
        self.args = args
        self.logger = logger
        self.train_loader = train_loader


    def train(self):
        raise NotImplementedError()


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