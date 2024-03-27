import numpy as np
import json
import torch
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random
import math
import torch.nn as nn

from utils import *
from client import Client, test,train, CustomDataset
from dataset import load_motionsense,load_wisdm, load_hhar,load_usc_had, load_uci_har
from model import init_weights, Conv1DModel

apply_transform = None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='motionsense', help='dataset used for training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--n_test_client', type=float, default=0.2, help='fraction of clients used as test (default:0.2)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--val_size', type=float, default=0.2, help='client self-evaluation dataest size( default: 0.2)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--personalized_epochs', type=int, default=0, help='number of personalization epochs')
    parser.add_argument('--unlabeled_portion', type=float, default=0.0, help='portion of unlabeld data')
    parser.add_argument('--max_samples', type=int, default=None, help='max number of samples to use for training')
    parser.add_argument('--max_num_drop_class', type=int, default=0, help='max number of dropped class for training')
    parser.add_argument('--alg', type=str, default='fedcoad',
                        help='communication strategy: fedavg/centralized/fedprox/fedcoad')
    parser.add_argument('--comm_round', type=int, default=100, help='number of maximum communication roun')
    parser.add_argument('--n_class', type=int, default=6, help='number of class available')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--loss', type=str, default='ce', help='loss function: ce, masked_ce')
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--device', type=str, default='cuda', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='adam', help='the optimizer')
    parser.add_argument('--normalization', type=str, default='pass', help='normalization option: [pass, zscore, minmax]')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or fedcoad')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--save_model',type=int,default=0)
    parser.add_argument('--threshold', type=float, default=0.8, help='threshold for pseudo-labelling')
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    parser.add_argument("--scheduler", action='store_true', help='use scheduler')
    parser.add_argument("--scheduler_epoch", type=int, default=60, help= 'when to reduce')
    args = parser.parse_args()
    return args

def schedule_learning_rate(lr, epoch, when_to_reduce=60):
    # apply lr scheduler to avoid gradient explode
    if epoch > when_to_reduce:
        lr = lr * math.exp(-0.1)
        print(f"learning rate reduced to:{lr}")
        logger.info(f"learning rate reduced to:{lr}")
    return lr

def create_model(n_class):
    return Conv1DModel(3, 96, n_class, use_head=True)

def load_dataset(dataset_name, mode='train'):
    subjects_dataset = []
    subjects_label = []
    if dataset_name == 'motionsense':
        subjects_dataset, subjects_label = load_motionsense(loc='../../dataset/Motionsense', freq=50, sec=4, channel_first=True)
    elif dataset_name == 'wisdm':
        subjects_dataset, subjects_label = load_wisdm(loc='../../dataset/WISDM_2011/WISDM_ar_v1.1_raw.txt', freq=20, sec=10, channel_first=True)
    elif dataset_name == 'usc_had':
        subjects_dataset, subjects_label = load_usc_had(loc='../../dataset/USC-HAD', freq=50, sec=4, channel_first=True)
    elif dataset_name == 'uci_har':
        subjects_dataset, subjects_label, _ = load_uci_har(loc='../../dataset/UCI_smartphone',mode=mode, channel_first=True)
    elif dataset_name == 'hhar':
        subjects_dataset, subjects_label = load_hhar(loc='../../dataset/HHAR/Activity recognition exp/Activity recognition exp', freq=50, sec=4, channel_first=True)
    else:
        print('dataset does not exist')
    return subjects_dataset, subjects_label

if __name__ == '__main__':
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path = f'{args.dataset}_{args.alg}_arguments-{datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")}.json'
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(vars(args), f, indent=4)
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = f'{args.dataset}_{args.alg}_log-{datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")}'
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    logger.info("Load data...")
    
    subjects_dataset, subjects_label = load_dataset(args.dataset)
        
    logger.info("partition client into test and train...")
    number_of_clients = len(subjects_label)
    
    if args.n_test_client > 0.0:
        if args.dataset == 'uci_har':
            #TODO: simplify this later
            _,_, test_clients_index = load_uci_har(loc='../../dataset/UCI_smartphone',mode='test', channel_first=True)
            _,_, train_clients_index = load_uci_har(loc='../../dataset/UCI_smartphone',mode='train', channel_first=True)
            test_clients_index= list(test_clients_index)
            train_clients_index = list(train_clients_index)
        else:
            number_of_test_clients = math.ceil(args.n_test_client*number_of_clients)
            test_clients_index = random.sample(list(range(0,number_of_clients)), number_of_test_clients)
            train_clients_index = list(set(list(range(number_of_clients))).difference(test_clients_index))
        
        logger.info(f"test clients index : {test_clients_index}")
        logger.info(f"train clients index : {train_clients_index}")
    else:
        train_clients_index = list(range(number_of_clients))
        test_clients_index = None
    
    # oarty that join per round
    n_party_per_round = int(len(train_clients_index) * args.sample_fraction)
    party_list_rounds = []
    if n_party_per_round != len(train_clients_index):
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(train_clients_index, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(train_clients_index)
            
    
    # initialize global model
    logger.info(f"initialize global model...")
    global_model = create_model(args.n_class)

    # init weights
    init_weights(global_model, 'xavier', 1.0)
    
    # initialize the clients
    logger.info(f"initialize training clients...")
    train_clients = {}
    for index,client_index in enumerate(train_clients_index):
        client = Client(
                client_config= {
                    "id":client_index,
                    "device":args.device,
                    "train_data": subjects_dataset[client_index] if args.dataset != 'uci_har' else subjects_dataset[index],
                    "train_label": subjects_label[client_index] if args.dataset != 'uci_har' else subjects_label[index],
                    "transform": apply_transform,
                    "batch_size": args.batch_size,
                    "val_size": args.val_size,
                    "max_samples": args.max_samples,
                    "unlabeled_portion": args.unlabeled_portion,
                    "max_num_drop_class": args.max_num_drop_class,
                    "num_global_class": args.n_class,
                    "normalization": args.normalization,
                    "threshold": args.threshold
                }
            )
        # initialize model
        #client.model = TCNModel(num_inputs=3, num_channels=[64,128,256], num_class=args.n_class, kernel_size=3)
        client.model = create_model(args.n_class)
        client.model.load_state_dict(global_model.state_dict())
        train_clients[client_index]=client
    
    # initialize server test dataset using test client dataset
    logger.info(f"initialize server data...")
    server_train_data = CustomDataset(
        np.concatenate([client.train_data for client in train_clients.values()], axis=0),
        np.concatenate([client.train_label for client in train_clients.values()], axis=0),
        transforms=None
    )
    
    # if there are no global testing dataset
    if test_clients_index == None:
        server_test_dataset = CustomDataset(
            np.concatenate([client.valid_data for client in train_clients.values()], axis=0),
            np.concatenate([client.valid_label for client in train_clients.values()], axis=0),
            transforms=None
        )  
    else:
        if args.dataset == 'uci_har':
            uci_test_data, uci_test_label = load_dataset(args.dataset,mode='test')
            server_test_dataset = CustomDataset(
                np.concatenate(uci_test_data, axis=0),
                np.concatenate(uci_test_label, axis=0),
                transforms=None
            )
        else:
            server_test_dataset = CustomDataset(
                np.concatenate([subjects_dataset[client_index] for client_index in test_clients_index], axis=0),
                np.concatenate([subjects_label[client_index] for client_index in test_clients_index], axis=0),
                transforms=None
            )
    server_train_dataloader =  torch.utils.data.DataLoader(server_train_data,batch_size=args.batch_size, drop_last=True)
    server_test_dataloader = torch.utils.data.DataLoader(server_test_dataset,batch_size=1)
    logger.info(f"number of server dataset:train=>{len(server_train_dataloader.sampler)} test=>{len(server_test_dataloader.sampler)}")
    
    n_comm_rounds = args.comm_round
    logger.info(f"pick training algorithm..")
    
    valid_loss_min = np.Inf
    best_f1 = 0.0
    metrics_recorder = AvgMeter()
    
    
    # classification loss function
    if args.loss == 'ce':
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss == 'masked_ce':
        loss_fn = LocalMaskCrossEntropyLoss(args.n_class)
    
    
    # pick training algorithm
    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0
                                
    if args.alg == 'fedcoad':
        logger.info(f"training algorithm: FedCoad")
        
        # initialize c_global
        c_global = create_model(args.n_class)
        c_global.eval()
        for param in c_global.parameters():
            param.requires_grad = False
        
        # initialize c_model
        for client in train_clients.values():
            #client.c_model = TCNModel(num_inputs=3, num_channels=[64,128,256], num_class=args.n_class, kernel_size=3)
            client.c_model =create_model(args.n_class)
            client.c_model.eval()
            for param in client.c_model.parameters():
                param.requires_grad = False
            client.c_model.load_state_dict(c_global.state_dict())
            
        # initialize previous model
        for client in train_clients.values():
            client.previous_model = create_model(args.n_class)
            client.previous_model.eval()
            for param in client.previous_model.parameters():
                param.requires_grad = False
            client.previous_model.load_state_dict(client.model.state_dict())
            
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            print("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]
            
            # initialize delta key
            total_delta = copy.deepcopy(global_model.state_dict())
            for key in total_delta:
                total_delta[key] = 0.0
                
            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False
            global_w = global_model.state_dict()
            
            clients_this_round = {k: train_clients[k] for k in party_list_rounds[round]}
            
            # update local model with global model
            train_results = []
            test_results = []
            
            # apply lr scheduler to avoid gradient explode
            if args.scheduler:
                args.lr = schedule_learning_rate(args.lr, round, args.scheduler_epoch)
            for client in clients_this_round.values():
                logger.info(f"local training client {client.id}..")
                print(f"local training client {client.id}..")
                client.model.load_state_dict(global_w)
                # do local training
                trainacc,testacc = client.local_train_net(args, global_model, loss_fn=loss_fn, round=round, c_global=c_global)
                train_results.append(trainacc)
                test_results.append(testacc)
                
                # update total_delta
                for key in total_delta:
                    total_delta[key] += client.delta_para[key]
            
            logger.info("done local training " + str(round))
            print("done local training "+ str(round))
            
            
            logger.info(f'Average train results: {dict_mean(train_results)}')
            logger.info(f'Average test results: {dict_mean(test_results)}')
            
            # update total_delta
            for key in total_delta:
                total_delta[key] /= len(train_clients_index)
            
            # update c_global    
            c_global_para = copy.deepcopy(c_global.state_dict())
            for key in c_global_para:
                if c_global_para[key].type() == 'torch.LongTensor':
                    c_global_para[key] += total_delta[key].type(torch.LongTensor)
                elif c_global_para[key].type() == 'torch.cuda.LongTensor':
                    c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
                else:
                    #print(c_global_para[key].type())
                    c_global_para[key] += total_delta[key].cpu()
            c_global.load_state_dict(c_global_para)

            # do averaging for the global model
            total_data_points = sum([len(client)for client in clients_this_round.values()])
            fed_avg_freqs = [len(client)/ total_data_points for client in clients_this_round.values()]

            for net_id, client in enumerate(clients_this_round.values()):
                net_para = client.model.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]
                        
             # update global model  
            global_model.load_state_dict(global_w)
            
            logger.info(f'global n_test: {len(server_test_dataloader.sampler)}')
            global_model.to(args.device)
            train_global_results, train_conf_matrix = test(global_model, server_train_dataloader, get_confusion_matrix=True, loss_fn=loss_fn, device=device)
            test_global_results, test_conf_matrix = test(global_model, server_test_dataloader, get_confusion_matrix=True, loss_fn=loss_fn, device=device)
            global_model.to('cpu')
            
            # save metrics
            metrics_recorder.save_metric(test_global_results ,dict_mean(test_results))
            
            logger.info(f">> Global Model Test results: {test_global_results}")
            print(f">> Global Model Test results: {test_global_results}")
            logger.info(f">> Global Model Train results: {train_global_results}")
            print(f">> Global Model Train results: {train_global_results}")
            
            # record best f1
            if best_f1< test_global_results["f1"]:
                best_f1 = test_global_results["f1"]
            
            valid_loss = test_global_results["loss"]
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
                torch.save(global_model.state_dict(), f'models/{args.dataset}_{args.alg}.pt')
                valid_loss_min = valid_loss
            
            logger.info("updating previous models")
            print("updating previous models")
            # update previous model
            for client in clients_this_round.values():
                client.previous_model.load_state_dict(client.model.state_dict())
                
    elif args.alg == 'fedavg':
        logger.info(f"training algorithm: FedAvg")
        
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            print("in comm round:" + str(round))
            
            party_list_this_round = party_list_rounds[round]

            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False
            global_w = global_model.state_dict()
            
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            clients_this_round = {k: train_clients[k] for k in party_list_rounds[round]}
            
            # update local model with global model
            train_results = []
            test_results = []
            # apply lr scheduler to avoid gradient explode
            if args.scheduler:
                args.lr = schedule_learning_rate(args.lr, round, args.scheduler_epoch)
            for client in clients_this_round.values():
                logger.info(f"local training client {client.id}..")
                print(f"local training client {client.id}..")
                client.model.load_state_dict(global_w)
                # do local training
                trainacc,testacc = client.local_train_net(args, global_model, loss_fn=loss_fn, round=round)
                train_results.append(trainacc)
                test_results.append(testacc)
            
            logger.info("done local training " + str(round))
            print("done local training "+ str(round))
            
            logger.info(f'Average train results: {dict_mean(train_results)}')
            logger.info(f'Average test results: {dict_mean(test_results)}')
            
            # do averaging for the global model
            total_data_points = sum([len(client)for client in clients_this_round.values()])
            fed_avg_freqs = [len(client)/ total_data_points for client in clients_this_round.values()]

            for net_id, client in enumerate(clients_this_round.values()):
                net_para = client.model.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]
                        
            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]
                    
            # update global model  
            global_model.load_state_dict(global_w)
            
            logger.info(f'global n_test: {len(server_test_dataloader.sampler)}')
            global_model.to(args.device)
            train_global_results, train_conf_matrix = test(global_model, server_train_dataloader, get_confusion_matrix=True, loss_fn=loss_fn, device=device)
            test_global_results, test_conf_matrix = test(global_model, server_test_dataloader, get_confusion_matrix=True, loss_fn=loss_fn, device=device)
            global_model.to('cpu')
            
            logger.info(f">> Global Model Test results: {test_global_results}")
            print(f">> Global Model Test results: {test_global_results}")
            logger.info(f">> Global Model Train results: {train_global_results}")
            print(f">> Global Model Train results: {train_global_results}")
            
            # save metrics
            metrics_recorder.save_metric(test_global_results ,dict_mean(test_results))
            # record best f1
            if best_f1< test_global_results["f1"]:
                best_f1 = test_global_results["f1"]
            
            valid_loss = test_global_results["loss"]
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
                torch.save(global_model.state_dict(), f'models/{args.dataset}_{args.alg}.pt')
                valid_loss_min = valid_loss
    
    elif args.alg == 'solo':
        logger.info(f"training algorithm: SOLO")
        
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            print("in comm round:" + str(round))
            
            party_list_this_round = party_list_rounds[round]
            
            # Just for passing it into the function
            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False
            global_w = global_model.state_dict()

            clients_this_round = {k: train_clients[k] for k in party_list_rounds[round]}
            
            # update local model with global model
            train_results = []
            test_results = []
            # apply lr scheduler to avoid gradient explode
            if args.scheduler:
                args.lr = schedule_learning_rate(args.lr, round, args.scheduler_epoch)
                
            for client in clients_this_round.values():
                logger.info(f"individual training client {client.id}..")
                print(f"individual training client {client.id}..")

                # do local training
                trainacc,testacc = client.local_train_net(args, global_model, loss_fn=loss_fn, round=round)
                train_results.append(trainacc)
                test_results.append(testacc)
            
            logger.info("done local training " + str(round))
            print("done local training "+ str(round))
            
            averaged_test_result = dict_mean(test_results)
            
            logger.info(f'Average train results: {dict_mean(train_results)}')
            logger.info(f'Average test results: {averaged_test_result}')
            
            # save metrics
            metrics_recorder.save_metric(fed_metrics=averaged_test_result)
            # record best f1
            if best_f1< averaged_test_result["f1"]:
                best_f1 = averaged_test_result["f1"]
            
                
    elif args.alg == 'fedprox':
        logger.info(f"training algorithm: FedProx")
        
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            print("in comm round:" + str(round))
            
            party_list_this_round = party_list_rounds[round]

            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False
            global_w = global_model.state_dict()
            
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            clients_this_round = {k: train_clients[k] for k in party_list_rounds[round]}
            
            # update local model with global model
            train_results = []
            test_results = []
            # apply lr scheduler to avoid gradient explode
            if args.scheduler:
                args.lr = schedule_learning_rate(args.lr, round, args.scheduler_epoch)
            for client in clients_this_round.values():
                logger.info(f"local training client {client.id}..")
                print(f"local training client {client.id}..")
                client.model.load_state_dict(global_w)
                # do local training
                trainacc,testacc = client.local_train_net(args, global_model, loss_fn=loss_fn, round=round)
                train_results.append(trainacc)
                test_results.append(testacc)
                
            logger.info("done local training " + str(round))
            print("done local training "+ str(round))
            
            logger.info(f'Average train results: {dict_mean(train_results)}')
            logger.info(f'Average test results: {dict_mean(test_results)}')
            
            # do averaging for the global model
            total_data_points = sum([len(client)for client in clients_this_round.values()])
            fed_avg_freqs = [len(client)/ total_data_points for client in clients_this_round.values()]

            for net_id, client in enumerate(clients_this_round.values()):
                net_para = client.model.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]
                        
             # update global model  
            global_model.load_state_dict(global_w)
            
            logger.info(f'global n_test: {len(server_test_dataloader.sampler)}')
            global_model.to(args.device)
            train_global_results, train_conf_matrix = test(global_model, server_train_dataloader, get_confusion_matrix=True, loss_fn= loss_fn, device=device)
            test_global_results, test_conf_matrix = test(global_model, server_test_dataloader, get_confusion_matrix=True, loss_fn=loss_fn, device=device)
            global_model.to('cpu')
            
            logger.info(f">> Global Model Test results: {test_global_results}")
            print(f">> Global Model Test results: {test_global_results}")
            logger.info(f">> Global Model Train results: {train_global_results}")
            print(f">> Global Model Train results: {train_global_results}")
            
            # save metrics
            metrics_recorder.save_metric(test_global_results ,dict_mean(test_results))
            # record best f1
            if best_f1< test_global_results["f1"]:
                best_f1 = test_global_results["f1"]
            
            valid_loss = test_global_results["loss"]
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
                torch.save(global_model.state_dict(), f'models/{args.dataset}_{args.alg}.pt')
                valid_loss_min = valid_loss
                
                
    # last federated validation result
    test_results = []
    for client_id, client in train_clients.items():
        if client.valid_loader is not None:
            
            test_client, _ = test(client.model, client.valid_loader, get_confusion_matrix=True, loss_fn=loss_fn, device=device)
            test_results.append(test_client)
            
    if len(test_results) > 0:        
        logger.info(f'Average test results: {dict_mean(test_results)}')
        metrics_recorder.save_metric(fed_metrics=dict_mean(test_results))
        
    # print metrics
    metrics_recorder.print_results()
    # print the best result
    logger.info(f">> Best achieved F1-score according to Global Test : {best_f1}")
    print(f">> Best achieved F1-score according to Global Test : {best_f1}")
    

            
            


 
    