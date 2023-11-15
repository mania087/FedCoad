import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

#########################
# Weight initialization #
#########################
def init_weights(model, init_type, init_gain):
    """Initialize network weights.

    Args:
        model (torch.nn.Module): network to be initialized
        init_type (string): the name of an initialization method: normal | xavier | xavier_uniform | kaiming | orthogonal | none
        init_gain (float): scaling factor for normal, xavier and orthogonal

    Returns:
        model (torch.nn.Module): initialized model with `init_type` and `init_gain`
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                torch.nn.init.normal_(m.weight.data, mean=1.0, std=init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, mean=0.0, std=init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'none':  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError(f'[ERROR] Initialization method {init_type} is not implemented!')
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
    model.apply(init_func)
    

class ConvolutionalEncoder(nn.Module):
    def __init__(self, num_channel, num_out=96):
        super(ConvolutionalEncoder, self).__init__()
        
        self.conv1 = nn.Conv1d(num_channel, 32, kernel_size=24)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=16)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        
        self.conv3 = nn.Conv1d(64, num_out, kernel_size=8)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.1)
        
        self.global_max_pooling = nn.AdaptiveMaxPool1d(1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        x = self.global_max_pooling(x)
        x = x.squeeze(2)
        
        return x
    
    
class Conv1DModel(nn.Module):
    def __init__(self, num_channel, num_out, num_class, hidden_layer=256, use_head=False):
        super(Conv1DModel, self).__init__()
        
        self.use_head = use_head
        
        self.encoder = ConvolutionalEncoder(num_channel,num_out)
        
        if self.use_head:
            self.head = nn.Sequential(
                nn.Linear(num_out,hidden_layer),
                nn.ReLU(),
                nn.Linear(hidden_layer,hidden_layer)
            )
            
            self.classifier = nn.Linear(hidden_layer,num_class)
            
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_out,hidden_layer),
                nn.ReLU(),
                nn.Linear(hidden_layer,num_class)
            )
        
        
    def forward(self, x):
        x = self.encoder(x)
        
        if self.use_head:
            y = self.head(x)
            out = self.classifier(y)
        else:
            y = 0.0
            out = self.classifier(x)
        
        multi_out = []
                
        return x, y, out, multi_out
