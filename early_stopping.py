import numpy as np
import torch


class EarlyStopping(object):
    """
    Early stops the training if validation loss doesn't decrease after a given patience.
    """
    def __init__(self, log_path, patience=7, verbose=False, delta=0):
        """
        log_path - absolute path, where the models (or parameter) is stored
        patience - int, how long to wait after last time validation loss decreased. Default: 7
        verbose  - bool, if True, prints a message for each validation loss improvement. Default: False
        delta    - float, minimum change in the monitored quantity to qualify as an improvement. Default: 0
        """
        # Preprocess the log_path
        log_path = log_path.strip()      # remove whitesapce before and after the txt
        log_path = log_path.rstrip("/")  # remove '/' at the end

        self.log_path = log_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

        # Color definition
        # self.RED_COLOR = "\033[31m"
        self.Green_COLOR = "\033[32m"
        self.BLUE_COLOR = "\033[34m"
        self.RESET_COLOR = "\033[0m"

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1

            print(f"{self.Green_COLOR}EarlyStopping counter: {self.counter} out of {self.patience}{self.RESET_COLOR}")

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Save models parameters when validation loss decrease.
        """
        if self.verbose:
            print(f"{self.BLUE_COLOR}Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving models parameters ...{self.RESET_COLOR}")
        # Save best models parameters
        torch.save(model.state_dict(), self.log_path + '/best_param.pth')
        self.val_loss_min = val_loss




