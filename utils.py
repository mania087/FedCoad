
import os
import logging
import torch
import matplotlib.pyplot as plt 
from torch import nn
import torch.nn.functional as F

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class LocalMaskCrossEntropyLoss(nn.CrossEntropyLoss):
    """Should be used for class-wise non-iid.
    Refer to HeteroFL (https://openreview.net/forum?id=TNkPBBYFkXg)
    """
    def __init__(self, num_classes, **kwargs):
        super(LocalMaskCrossEntropyLoss, self).__init__(**kwargs)
        self.num_classes = num_classes
        
    def forward(self, input, target):
        classes = torch.unique(target)
        mask = torch.zeros_like(input)
        for c in range(self.num_classes):
            if c in classes:
                mask[:, c] = 1  # select included classes
        return F.cross_entropy(input*mask, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)
        
class AvgMeter():
    def __init__(self):
        self.client_tests = {
            "loss": [],
            "acc": [],
            "prec": [],
            "rec": [],
            "f1": [],
        }
        self.global_tests = {
            "loss": [],
            "acc": [],
            "prec": [],
            "rec": [],
            "f1": [],
        }
    
    def save_metric(self, global_metrics = None, fed_metrics = None):
        
        for key in self.client_tests.keys():
            if global_metrics is not None:
                self.global_tests[key].append(global_metrics[key])
            if fed_metrics is not None:
                self.client_tests[key].append(fed_metrics[key])
            
    def print_results(self):
        print(f"Federated testing results : {self.client_tests}")
        print(f"Global testing results : {self.global_tests}")
        logger.info(f"Federated testing results : {self.client_tests}")
        logger.info(f"Global testing results : {self.global_tests}")
        
        # to avoid error of empty list
        for key in self.client_tests.keys():
            self.global_tests[key].append(0)
            self.client_tests[key].append(0)
        
        print("Federated testing:")
        logger.info("Federated testing:")
        for key in self.client_tests.keys():
            print(f"Maximum {key}: {max(self.client_tests[key])}")
            logger.info(f"Maximum {key}: {max(self.client_tests[key])}")
            
        print("Global testing:")
        logger.info("Global testing:")
        for key in self.global_tests.keys():
            print(f"Maximum {key}: {max(self.global_tests[key])}")
            logger.info(f"Maximum {key}: {max(self.global_tests[key])}")
        
def plot_signal(signal):
    # plot lines
    plt.plot(signal[0], label = "X signal")
    plt.plot(signal[1], label = "Y signal")
    plt.plot(signal[2], label = "Z signal")
    plt.legend()
    plt.show()
    
def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass
    
