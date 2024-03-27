import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import copy
import random
import torch.optim as optim

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix
from torch.utils.data.sampler import SubsetRandomSampler

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class CustomDataset(Dataset):
    def __init__(self, X, y, transforms=None):
        self.X = X
        self.y = y
        self.transforms = transforms

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]

        if self.transforms is not None:
            x = np.expand_dims(x, axis=0)
            x = self.transforms(x)
            x = np.squeeze(x, axis=0)
        
        #convert to tensor
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)

        return x,y
    
# create limited labeled
def drop_class(X, y, classes_to_drop):
    pick_indexes = []
    for i in classes_to_drop:
        # pick the index where the class is i
        # check if samples are enough
        index_with_label = np.where(y==i)[0]
        pick_indexes.extend(index_with_label)
    # delete the data according to index
    new_X = np.delete(X, pick_indexes, axis=0)
    new_y = np.delete(y, pick_indexes, axis=0)
    
    return new_X, new_y

def create_limited_labeled_data(X, y, classes, num_samples):
    pick_indexes = []
    for i in classes:
        # pick the index where the class is i
        # check if samples are enough
        if len(np.where(y==i)[0]) >= num_samples:
            pick_indexes.extend(np.random.choice(np.where(y==i)[0],num_samples,replace=False))
        else:
            pick_indexes.extend(np.where(y==i)[0])
    new_X = X[pick_indexes]
    new_y = y[pick_indexes]
    
    return new_X, new_y

def test(net, testloader, get_confusion_matrix=False, loss_fn = nn.CrossEntropyLoss(), device: str = "cpu"):
    """Validate the network on the entire test set."""
    criterion = loss_fn
    loss = 0.0
    
    net.to(device)
    net.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(testloader):
            data, labels = x.to(device), y.to(device)
            _, _, outputs, _ = net(data)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            # appending
            y_pred.extend(predicted.cpu().detach().numpy())
            y_true.extend(labels.cpu().detach().numpy())
    
    loss = loss / len(testloader)
    net.to("cpu")  # move model back to CPU
    
    # convert tensors to numpy arrays
    y_true = np.array(y_true,dtype=np.int64)
    y_pred = np.array(y_pred,dtype=np.int64)

    # calculate accuracy
    acc = accuracy_score(y_true, y_pred)
    # calculate precision
    precision = precision_score(y_true, y_pred, average='macro')
    # calculate recall
    recall = recall_score(y_true, y_pred, average='macro')
    # calculate F1-score
    f1 = f1_score(y_true, y_pred, average='macro')

    results = {
        "loss": loss,
        "acc":acc,
        "prec":precision,
        "rec":recall,
        "f1":f1,
    }
    
    if get_confusion_matrix:
        conf_matrix = confusion_matrix(y_true, y_pred)
        return results, conf_matrix
    else:
        return results 

def train(net, trainloader, lr, args_optimizer, args, round, loss_fn = nn.CrossEntropyLoss(), device="cpu"):
    
    logger.info('n_training: %d' % len(trainloader.sampler))
        
    criterion= loss_fn
    train_loss = 0.0
    
    net.to(device)      
    net.train()
    
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                            amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                            weight_decay=args.reg)
        
    for batch_idx, (x,target) in enumerate(trainloader):
        x, target = x.to(device), target.to(device)
        #print(torch.isnan(x).any())
        
        optimizer.zero_grad()
        target = target.long()
        
        _, _, out, _ = net(x)
        
        loss = criterion(out, target)
        
        loss.backward()
        
        # clipping
        #torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        
        optimizer.step()
        
        train_loss += loss.item()*x.size(0)
        
    train_loss = train_loss/len(trainloader.sampler)
    logger.info('Epoch: %d Loss: %f' % (round, train_loss))
    
    train_acc = test(net, trainloader, device=device)
    logger.info('>> After-Training Training accuracy: {}'.format(train_acc))
    
    net.to('cpu')

def normalize_data(norm_method, X_train, X_test=None):
    # if we do normalization
    normalized_test = []
    if norm_method == 'minmax':
        channel_max = np.max(X_train, axis=(0,2)).reshape((1, X_train.shape[1], 1))
        channel_min = np.min(X_train, axis=(0,2)).reshape((1, X_train.shape[1], 1))
        normalized_train = (X_train  - channel_min) / (channel_max - channel_min)
        if X_test is not None:
            normalized_test = (X_test  - channel_min) / (channel_max - channel_min)
            
    elif norm_method == 'zscore':
        channel_mean = np.mean(X_train, axis=(0,2)).reshape((1, X_train.shape[1], 1))
        channel_std = np.std(X_train, axis=(0,2)).reshape((1, X_train.shape[1], 1))
        normalized_train = (X_train - channel_mean) / channel_std
        if X_test is not None:
            normalized_test = (X_test - channel_mean) / channel_std
    else:
        normalized_train = X_train
        if X_test is not None:
            normalized_test = X_test
           
    if X_test is not None:
        return normalized_train, normalized_test
    else:
        return normalized_train
    
class Client():
    def __init__(self, client_config:dict):
        # client config as dict to make configuration dynamic
        # you can call with:
        # Client(client_config={
        #    "id": client id,
        #    "device": device used to train
        #    "train_data": Tensordataset consisting of feature and label
        #    "batch_size": the batch size    
        #})
        self.id = client_config["id"]
        self.config = client_config
        self.__model = None
        self.device = client_config["device"]
        
        
        # prepare data loaders 
        # if we have consider limited_labeled_data

        # initial data info
        self.train_data = np.array(self.config["train_data"])
        self.train_label = np.array(self.config["train_label"])
        self.available_class = np.unique(self.train_label)
        
        self.valid_data = None
        self.valid_label = None
        logger.info(f"client {self.id} Starting available classes:{self.available_class }")
        
        # if we use validation
        if self.config["val_size"] > 0.0:
            
            X_train, X_valid, y_train, y_valid= train_test_split(self.train_data , self.train_label , 
                                                                 test_size=self.config["val_size"],
                                                                 stratify=self.train_label, random_state=42)
            
            # if there are normalization
            logger.info(f"client {self.id} normalize data using {self.config['normalization']} ...")
            self.train_data, self.valid_data  = normalize_data(self.config["normalization"], X_train, X_valid)
            
            # update from index partition
            self.train_label = y_train
            self.available_class = np.unique(self.train_label)
            
            self.valid_label = y_valid
            
            
            valid_dataset = CustomDataset(self.valid_data,
                                          self.valid_label,
                                          transforms= self.config['transform']
                                          )
            
            train_dataset = CustomDataset(self.train_data,
                                          self.train_label,
                                          transforms= self.config['transform']
                                          )
            
            self.train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                            batch_size=self.config["batch_size"],
                                                            drop_last=True)
            
            self.valid_loader = torch.utils.data.DataLoader(valid_dataset, 
                                                            batch_size=self.config["batch_size"])
            
            logger.info(f"client {self.id} number of testing data:{len(self.valid_loader.sampler)}")
        else:
            # create train torch dataset
            logger.info(f"client {self.id} normalize data using {self.config['normalization']} ...")
            self.train_data = normalize_data(self.config["normalization"], self.train_data)
            
            client_dataset = CustomDataset(self.train_data,
                                           self.train_label,
                                           transforms= self.config['transform']
                                           ) 
            
            self.train_loader = torch.utils.data.DataLoader(client_dataset, 
                                                            batch_size=self.config["batch_size"],
                                                            drop_last=True)
            self.valid_loader = None
            logger.info(f"client {self.id} number of testing data: 0")
            
        # if there are unlabeled data partition
        if self.config["unlabeled_portion"] > 0.0:
            X_train, X_unlabeled, y_train, y_unlabeled = train_test_split(self.train_data , self.train_label , 
                                                                          test_size=self.config["unlabeled_portion"],
                                                                          stratify=self.train_label, random_state=42)
            # update from index partition
            self.train_data = X_train
            self.train_label = y_train
            self.available_class = np.unique(self.train_label)
            
            # unlabeled data
            self.unlabeled_data = X_unlabeled
            
            # create dataloader
            unlabeled_dataset = CustomDataset(X_unlabeled,
                                              y_unlabeled,
                                              transforms= self.config['transform']
                                              )
            train_dataset = CustomDataset(X_train,
                                          y_train,
                                          transforms= self.config['transform']
                                          )
            
            self.train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                            batch_size=self.config["batch_size"],
                                                            drop_last=True)
            
            self.unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, 
                                                                batch_size=self.config["batch_size"])
            
            logger.info(f"client {self.id} number of unlabeled data: {len(self.unlabeled_loader.sampler)}") 
        
            
        # if limited number of data
        if self.config["max_samples"] is not None:
            print(f"use limited number of labeled scheme:{self.config['max_samples']}")

            
            new_X, new_y = create_limited_labeled_data(self.train_data, self.train_label, self.available_class, self.config["max_samples"])
            
            # update current data state
            self.train_data = new_X
            self.train_label = new_y
            self.available_class = np.unique(self.train_label)
            
            client_train_data = CustomDataset(new_X, 
                                              new_y,
                                              transforms= self.config['transform']
                                              )
                    
            self.train_loader = torch.utils.data.DataLoader(client_train_data, 
                                                            batch_size=self.config["batch_size"],
                                                            shuffle=True)
        # if dropping class
        if (self.config["max_num_drop_class"] > 0) and (len(self.available_class) > self.config["max_num_drop_class"]):
            
            number_of_class_to_drop = random.randint(0, self.config["max_num_drop_class"])
            
            if number_of_class_to_drop > 0:
                pick_random_class = random.sample(list(self.available_class), number_of_class_to_drop)
                print(f"client {self.id} dropping:{pick_random_class}")
                logger.info(f"client {self.id} dropping:{pick_random_class}")
                
                new_X, new_y = drop_class(self.train_data, self.train_label, pick_random_class)
                
                # update current data state
                self.train_data = new_X
                self.train_label = new_y
                self.available_class = np.unique(self.train_label)
                
                client_train_data = CustomDataset(new_X, 
                                                  new_y,
                                                  transforms= self.config['transform']
                                                  )
                self.train_loader = torch.utils.data.DataLoader(client_train_data, 
                                                                batch_size=self.config["batch_size"],
                                                                shuffle=True)
                
        class_counter = {}
        for cls_index in range(self.config["num_global_class"]):
            class_counter[cls_index] = np.count_nonzero(self.train_label == cls_index)
        
        logger.info(f"client {self.id} training data:{class_counter}")
        logger.info(f"client {self.id} number of training data:{len(self.train_loader.sampler)}")
        
        # if add data augmentation
        
        
        # for contrastive learning algorithm
        self.previous_model = None
        # for control variates
        self.c_model = None
        self.delta_para = None
        
        # pseudo-label array
        self.x_pseudo = []
        self.y_pseudo = []

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.train_loader.sampler)
    
    def local_train_net(self, args, global_model, loss_fn = nn.CrossEntropyLoss(), round=None, c_global=None):
        
        if global_model:
            global_model.to(self.device)
        if c_global is not None:
            c_global.to(self.device)
            
        logger.info(f"Training network {self.id} n_training: {len(self.train_loader.sampler)}")

        n_epoch = args.epochs

        if args.alg == 'fedavg':
            trainacc, testacc = self.train_net_fedavg(n_epoch, args.lr, args.optimizer, loss_fn, args, device=self.device)
        elif args.alg == 'solo':
            # the difference with fedavg is we train using epoch=1
            trainacc, testacc = self.train_net_fedavg(1, args.lr, args.optimizer, loss_fn, args, device=self.device)
        elif args.alg == 'fedprox':
            trainacc, testacc = self.train_net_fedprox(global_model, n_epoch, args.lr,args.optimizer, args.mu, loss_fn, args, device=self.device)
        elif args.alg == 'fedcoad':
            trainacc, testacc = self.train_net_fedcoad(global_model, c_global, n_epoch, args.lr,args.optimizer, args.mu, args.temperature, loss_fn, args, round, device=self.device)
        
        logger.info(f"Client {self.id} final test {testacc}")

        if global_model:
            global_model.to('cpu')
        if c_global is not None:
            c_global.to('cpu')
            
        return trainacc,testacc
    
    def train_net_fedprox(self, global_model, epochs, lr, args_optimizer, mu, loss_fn, args, device="cpu"):
        logger.info('n_training: %d' % len(self.train_loader.sampler))
        train_acc = test(self.model, self.train_loader, device=device)
        
        if self.valid_loader is not None:
            logger.info('n_test: %d' %len(self.valid_loader.sampler))
            test_acc = test(self.model, self.valid_loader, device=device)
            logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))
            
        logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
        
        self.model.to(device)
        global_model.to(device)
        
        self.model.train()
        
        if args_optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=args.reg)
        elif args_optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=args.reg,
                                amsgrad=True)
        elif args_optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, momentum=0.9,
                                weight_decay=args.reg)
            
        criterion = loss_fn
        global_weight_collector = list(global_model.parameters())
        
        for epoch in range(epochs):
            epoch_loss_collector = []
            for batch_idx, (x, target) in enumerate(self.train_loader):
                x, target = x.to(device), target.to(device)
                optimizer.zero_grad()
                target = target.long()
        
                _, _, out, _ = self.model(x)
                loss = criterion(out, target)
                
                # fedprox requirement
                fed_prox_reg = 0.0
                for param_index, param in enumerate(self.model.parameters()):
                    fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss += fed_prox_reg
                
                loss.backward()
                optimizer.step()  
                
                epoch_loss_collector.append(loss.item())
                
            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
            
        # after training 
        train_acc = test(self.model, self.train_loader, device=device)
        
        if self.valid_loader:
            logger.info('n_test: %d' %len(self.valid_loader.sampler))
            test_acc = test(self.model, self.valid_loader, device=device)
            logger.info('>> After-Training Test accuracy: {}'.format(test_acc))
            
        logger.info('>> After-Training Training accuracy: {}'.format(train_acc))
        
        # if there are evaluations
        self.model.to('cpu')
        global_model.to('cpu')
        
        return train_acc,test_acc
        
    def train_net_fedavg(self, epochs, lr, args_optimizer, loss_fn, args, specified_data_loader = None, device="cpu"):
        
        if specified_data_loader is None:
            specified_data_loader = self.train_loader
        
        logger.info('n_training: %d' % len(specified_data_loader.sampler))
        train_acc = test(self.model, specified_data_loader, device=device)
        
        if self.valid_loader is not None:
            logger.info('n_test: %d' %len(self.valid_loader.sampler))
            test_acc = test(self.model, self.valid_loader, device=device)
            logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))
            
        logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
        
        self.model.to(device)
        self.model.train()
        
        if args_optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=args.reg)
        elif args_optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=args.reg,
                                amsgrad=True)
        elif args_optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, momentum=0.9,
                                weight_decay=args.reg)
            

        criterion = loss_fn
        
        
        for epoch in range(epochs):
            epoch_loss_collector = []
            for batch_idx, (x, target) in enumerate(specified_data_loader):
                x, target = x.to(device), target.to(device)
                optimizer.zero_grad()
                target = target.long()
                
                _, _, out, _ = self.model(x)
                
                loss = criterion(out, target)
                
                loss.backward()
                optimizer.step()

                epoch_loss_collector.append(loss.item())
                
            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
            
        # after training 
        train_acc = test(self.model, specified_data_loader, device=device)
        
        if self.valid_loader:
            logger.info('n_test: %d' %len(self.valid_loader.sampler))
            test_acc = test(self.model, self.valid_loader, device=device)
            logger.info('>> After-Training Test accuracy: {}'.format(test_acc))
            
        logger.info('>> After-Training Training accuracy: {}'.format(train_acc))
        
        # if there are evaluations
        self.model.to('cpu')
        
        return train_acc,test_acc
            
    def train_net_fedcoad(self, global_model, c_global, epochs, lr, args_optimizer, mu, temperature, loss_fn, args, round, device="cpu"):
        logger.info('n_training: %d' % len(self.train_loader.sampler))
        train_acc = test(self.model, self.train_loader, device=device)
        
        if self.valid_loader is not None:
            logger.info('n_test: %d' %len(self.valid_loader.sampler))
            test_acc = test(self.model, self.valid_loader, device=device)
            logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))
            
        logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
        
        self.model.to(device)
        self.previous_model.to(device)
        global_model.to(device)
        self.c_model.to(device)
        c_global.to(device)
        
        self.model.train()

        if args_optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=args.reg)
        elif args_optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=args.reg,
                                amsgrad=True)
        elif args_optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, momentum=0.9,
                                weight_decay=args.reg)

        criterion = loss_fn
        cos=torch.nn.CosineSimilarity(dim=-1)
        # mu = 0.001
        
        c_global_para = c_global.state_dict()
        c_local_para = self.c_model.state_dict()
        
        cnt = 0

        for epoch in range(epochs):
            epoch_loss_collector = []
            epoch_loss1_collector = []
            epoch_loss2_collector = []
            for batch_idx, (x, target) in enumerate(self.train_loader):
                x, target = x.to(device), target.to(device)
                optimizer.zero_grad()
                #x.requires_grad = False
                #target.requires_grad = False
                target = target.long()

                _, pro1, out, _ = self.model(x)
                _, pro2, _, _ = global_model(x)

                posi = cos(pro1, pro2)
                logits = posi.reshape(-1,1)

                _, pro3, _, _ = self.previous_model(x)
                nega = cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

                logits /= temperature
                labels = torch.zeros(x.size(0)).cuda().long()

                loss2 = mu * criterion(logits, labels)

                loss1 = criterion(out, target)
                loss = loss1 + loss2

                loss.backward()
                optimizer.step()
                
                net_para = self.model.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - args.lr * (c_global_para[key] - c_local_para[key])
                self.model.load_state_dict(net_para)

                cnt += 1

                epoch_loss_collector.append(loss.item())
                epoch_loss1_collector.append(loss1.item())
                epoch_loss2_collector.append(loss2.item())

            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
            epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
            logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))
        
        # update c_model
        c_new_para = self.c_model.state_dict()
        self.delta_para = copy.deepcopy(self.c_model.state_dict())
        global_model_para = global_model.state_dict()
        net_para = self.model.state_dict()
        
        for key in net_para:
            c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (cnt * args.lr)
            self.delta_para[key] = c_new_para[key] - c_local_para[key]
        self.c_model.load_state_dict(c_new_para)
        
        # after training 
        train_acc = test(self.model, self.train_loader, device=device)
        
        if self.valid_loader:
            logger.info('n_test: %d' %len(self.valid_loader.sampler))
            test_acc = test(self.model, self.valid_loader, device=device)
            logger.info('>> After-Training Test accuracy: {}'.format(test_acc))
            
        logger.info('>> After-Training Training accuracy: {}'.format(train_acc))
        
        # if there are evaluations
        self.model.to('cpu')
        self.previous_model.to('cpu')
        global_model.to('cpu')
        self.c_model.to('cpu')
        c_global.to("cpu")
        
        return train_acc,test_acc

