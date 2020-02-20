"""
Differentiable Physics-informed Graph Networks

Requirements:
    Python=3.6
    PyTorch>=0.4
    PyTorch Geometric
    PyTorch Scatter
    PyTorch Sparse
    
Usage:
    $ python main.py
"""
from __future__ import division
from __future__ import print_function

import sys
import os
import logging
import pprint
# import socket
import datetime

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.data import Data

from utils import get_laplacian
from model import Net


TIME_DIM = 384
LAT_DIM = 141    # vertical
LONG_DIM = 129   # horizontal


def main(cfg):

    MODE = cfg['model']['MODE']    # 1: GN-only, 2: Physics-only, 3: DPGN
    MODE_DESC = cfg['model']['MODE_desc']
    REGION = cfg['dataset']['REGION']    # LA or SD
    PDE = cfg['model']['PDE']
    NN = cfg['model']['NN']
    USE_INPUT_PRED_LOSS = cfg['model']['USE_INPUT_PRED_LOSS']
    pred_input_weight = cfg['model']['pred_input_weight']
    SKIP = cfg['model']['SKIP']

    dirname = "_".join([MODE_DESC, REGION, "NN"+str(NN), datetime.datetime.now().isoformat()])
    logdir = os.path.join("log", dirname)
    modeldir = os.path.join("model", dirname)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    logfilename = os.path.join(logdir, 'log.txt')
    
    # Print the configuration - just to make sure that you loaded what you wanted to load
    with open(logfilename, 'w') as f:
        pp = pprint.PrettyPrinter(indent=4, stream=f)
        pp.pprint(cfg)
    
    logging.basicConfig(filename=logfilename,
                        filemode='a',
                        format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.DEBUG)
    writer = SummaryWriter(logdir)
    
    logging.info("MODE: {} ({})\tREGION: {}".format(MODE, MODE_DESC, REGION))
    logging.info("logdir: {}".format(logdir))
    logging.info("modeldir: {}".format(modeldir))

    ########## Load data and edge attributes ##########
    X = np.load(cfg['dataset']['X_path'])
    edge_index = np.load(cfg['dataset']['edge_index_path'])
    edge_attr = np.load(cfg['dataset']['edge_attr_path'])

    edge_index = torch.tensor(edge_index)
    edge_attr = torch.tensor(edge_attr)
    num_nodes = X.shape[1]
    ###################################################


    ########## Device setting ##########
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ####################################


    ########## Architecture setting ##########
    node_attr_size = X.shape[2] - 1    # Temperature is not considered as input.
#     edge_attr_size = 1    # embedding index
    edge_num_embeddings = torch.max(edge_attr).item() + 2
    edge_hidden_size = cfg['model']['edge_dim']
    node_hidden_size = cfg['model']['node_dim']
    global_hidden_size = cfg['model']['global_dim']
    output_size = 1    # predict Temperature
    D = torch.tensor(cfg['model']['diff']).to(device)
    sp_L = get_laplacian(edge_index, type="norm").to(device)
    ##########################################

    num_processing_steps = cfg['train']['num_processing_steps']    # Forecast horizon
    num_iterations = cfg['train']['num_iter']

    losses_sup = []    # supervised loss
    losses_phy = []    # physics loss
    losses_tot = []    # total loss
    val_losses_sup = []
    used_timestamps = []

    #### Model ####
    if cfg['modelpath']:
        model = torch.load(cfg['modelpath'])
        logging.info("pretrained model is loaded. {}".format(cfg['modelpath']))
    else:
        model = Net(node_attr_size,
                    edge_num_embeddings, 
                    output_size, 
                    edge_hidden_size=edge_hidden_size, 
                    node_hidden_size=node_hidden_size, 
                    global_hidden_size=global_hidden_size,
                    skip=SKIP,
                    device=device)
        logging.info("new model is initialized. {}".format(modeldir))
        logging.info("random coefficients: {}, {}".format(model.gn.a, model.gn.b))

    num_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("# params in model: {}".format(num_total_params))

    model.to(device)
    model.train()

    # Training loss
    criterion_mse = nn.MSELoss()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), 
                           lr=cfg['optimizer']['initial_lr'], 
                           weight_decay=cfg['optimizer']['weight_decay'])
    reg_coeff = cfg['train']['reg_coeff']

    tr_ind, val_ind, te_ind = 250, 300, TIME_DIM-1    # training/validation/test split

    #### Training
    for iter_ in range(num_iterations):
        #### Sample a starting timestep randomly
        t = np.random.randint(0, tr_ind - num_processing_steps)    # low (inclusive) to high (exclusive)
        used_timestamps.append(t)

        #### Set input_graph
        # initial global_attr is dummy tensor.
        input_graphs = [Data(x=torch.tensor(X[t+step_t,:,1:], dtype=torch.float32, device=device), 
                             edge_index=edge_index.to(device), edge_attr=edge_attr.to(device))
                        for step_t in range(num_processing_steps)]
        input_graphs[0].global_attr = torch.zeros((1, global_hidden_size), device=device)    # initial global_attr

        #### Passing the model
#         output_tensors, time_derivatives, spatial_derivatives, _ = model(input_graphs, sp_L, num_processing_steps, D, PDE)
        output_tensors, time_derivatives, spatial_derivatives, pred_inputs = model(input_graphs, sp_L, num_processing_steps, D, PDE)

        #### Training loss across processing steps.
        loss_sup_seq = [torch.sum((output - torch.tensor(X[t+1+step_t,:,:1], dtype=torch.float32, device=device))**2)
                        for step_t, output in enumerate(output_tensors)]
        loss_sup = sum(loss_sup_seq) / len(loss_sup_seq)    # mean over num_predicted_steps

        #### Physics rule
        loss_phy_seq = [torch.sum((dt-ds)**2) for dt, ds in zip(time_derivatives, spatial_derivatives)]
        loss_phy = sum(loss_phy_seq) / len(loss_phy_seq)
        
        if USE_INPUT_PRED_LOSS:
            #### Use pred_inputs for optimization
            loss_pred_inputs_seq = [torch.sum((pred_input - torch.tensor(X[t+1+step_t,:,1:], dtype=torch.float32, device=device))**2)
                                    for step_t, pred_input in enumerate(pred_inputs)]
            loss_pred_inputs = sum(loss_pred_inputs_seq) / len(loss_pred_inputs_seq)
        
            loss_sup = loss_sup + pred_input_weight*loss_pred_inputs
        

        #### loss
        if MODE == 1:
            loss = loss_sup
        elif MODE == 2:
            loss = loss_phy
        elif MODE == 3:
            loss = loss_sup + reg_coeff*loss_phy

        losses_sup.append(loss_sup.item())
        losses_phy.append(loss_phy.item())
        losses_tot.append(loss.item())
        
        writer.add_scalars('loss/train', {'loss_sup': losses_sup[-1]}, iter_)
        writer.add_scalars('loss/train', {'loss_sup_per_node': losses_sup[-1]/num_nodes}, iter_)
        writer.add_scalars('loss/train', {'loss_phy': losses_phy[-1]}, iter_)
        writer.add_scalars('loss/train', {'loss_phy_per_node': losses_phy[-1]/num_nodes}, iter_)
        writer.add_scalars('loss/train', {'loss_tot': losses_tot[-1]}, iter_)
        writer.add_scalars('loss/train', {'loss_tot_per_node': losses_tot[-1]/num_nodes}, iter_)

        #### Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_ == 0:
            if MODE == 1:
                torch.save(model, os.path.join(modeldir, "supervised_only_model"))
            elif MODE == 2:
                torch.save(model, os.path.join(modeldir, "physics_only_model"))
            elif MODE == 3:
                torch.save(model, os.path.join(modeldir, "hybrid_model"))
                
        #### Validation
        if iter_%cfg['train']['valid_iter'] == 0:
            losses_val = []
            for vt in range(tr_ind, val_ind - num_processing_steps):

                input_graphs = [Data(x=torch.tensor(X[vt+step_t,:,1:], dtype=torch.float32, device=device), 
                                     edge_index=edge_index.to(device), edge_attr=edge_attr.to(device)) 
                                for step_t in range(num_processing_steps)]
                input_graphs[0].global_attr = torch.zeros((1, global_hidden_size), device=device)

                output_tensors, _, _, _ = model(input_graphs, sp_L, num_processing_steps, D)

                #### Validation loss across processing steps.
                val_loss_sup_seq = [torch.sum((output - torch.tensor(X[vt+1+step_t,:,:1], dtype=torch.float32, device=device))**2) 
                                    for step_t, output in enumerate(output_tensors)]
                val_loss_sup = sum(val_loss_sup_seq) / len(val_loss_sup_seq)    # mean over num_predicted_steps
                losses_val.append(val_loss_sup.item())

            if (len(val_losses_sup)>0) and (np.mean(losses_val)<np.min(val_losses_sup)):
                if MODE == 1:
                    torch.save(model, os.path.join(modeldir, "supervised_only_model"))
                elif MODE == 2:
                    torch.save(model, os.path.join(modeldir, "physics_only_model"))
                elif MODE == 3:
                    torch.save(model, os.path.join(modeldir, "hybrid_model"))
                    
                # When best validation is found, check test set
                losses_te = []
                for tt in range(val_ind, te_ind - num_processing_steps):

                    input_graphs = [Data(x=torch.tensor(X[tt+step_t,:,1:], dtype=torch.float32, device=device), 
                                         edge_index=edge_index.to(device), edge_attr=edge_attr.to(device))
                                    for step_t in range(num_processing_steps)]
                    input_graphs[0].global_attr = torch.zeros((1, global_hidden_size), device=device)

                    output_tensors, _, _, _ = model(input_graphs, sp_L, num_processing_steps, D)

                    #### Test loss across processing steps.
                    te_loss_sup_seq = [torch.sum((output - torch.tensor(X[tt+1+step_t,:,:1], dtype=torch.float32, device=device))**2) 
                                       for step_t, output in enumerate(output_tensors)]
                    te_loss_sup = sum(te_loss_sup_seq) / len(te_loss_sup_seq)

                    losses_te.append(te_loss_sup.item())
                    
                writer.add_scalars('loss/test', {'loss_sup': np.mean(losses_te)}, iter_)
                writer.add_scalars('loss/test', {'loss_sup_per_node': np.mean(losses_te)/num_nodes}, iter_)
                logging.info("{}/{} iterations.".format(iter_, num_iterations))
                logging.info("[Train]Loss: {:.4f}\tLoss_sup: {:.4f}\tLoss_phy: {:.4f}\t[Vali]Loss_sup: {:.4f}({:.4f})\t[Test]Loss_sup: {:.4f}({:.4f})"
                             .format(loss, loss_sup.item(), loss_phy.item(), 
                                     np.mean(losses_val), np.mean(losses_val)/num_nodes, 
                                     np.mean(losses_te), np.mean(losses_te)/num_nodes))

            val_losses_sup.append(np.mean(losses_val))
            writer.add_scalars('loss/valid', {'loss_sup': val_losses_sup[-1]}, iter_)
            writer.add_scalars('loss/valid', {'loss_sup_per_node': val_losses_sup[-1]/num_nodes}, iter_)

            
            
        if iter_%cfg['train']['verbose_iter'] == 0:
            logging.info("{}/{} iterations.".format(iter_, num_iterations))
            logging.info("[Train]Loss: {:.4f}\tLoss_sup: {:.4f}\tLoss_phy: {:.4f}\t[Vali]Loss_sup: {:.4f}({:.4f})"
                         .format(loss, loss_sup.item(), loss_phy.item(), np.mean(losses_val), np.mean(losses_val)/num_nodes))

    logging.info("[Training]The smallest supervised loss: {:.4e}({:.4e}) at {}/{}"
                 .format(np.min(losses_sup), np.min(losses_sup)/num_nodes, np.argmin(losses_sup), len(losses_sup)))
    logging.info("[Vali]The smallest supervised loss: {:.4e}({:.4e}) at {}/{}"
                 .format(np.min(val_losses_sup), np.min(val_losses_sup)/num_nodes, np.argmin(val_losses_sup), len(val_losses_sup)))


    """
    Final Test
    """
    if MODE == 1:
        model = torch.load(os.path.join(modeldir, "supervised_only_model"))
    elif MODE == 2:
        model = torch.load(os.path.join(modeldir, "physics_only_model"))
    elif MODE == 3:
        model = torch.load(os.path.join(modeldir, "hybrid_model"))

    model.eval()
    losses_te = []
    for tt in range(val_ind, te_ind - num_processing_steps):

        input_graphs = [Data(x=torch.tensor(X[tt+step_t,:,1:], dtype=torch.float32, device=device), 
                             edge_index=edge_index.to(device), edge_attr=edge_attr.to(device))
                        for step_t in range(num_processing_steps)]
        input_graphs[0].global_attr = torch.zeros((1, global_hidden_size), device=device)

        output_tensors, _, _, _ = model(input_graphs, sp_L, num_processing_steps, D)

        # Training loss across processing steps.
        loss_sup_seq = [torch.sum((output - torch.tensor(X[tt+1+step_t,:,:1], dtype=torch.float32, device=device))**2) 
                        for step_t, output in enumerate(output_tensors)]
        loss_sup = sum(loss_sup_seq) / len(loss_sup_seq)

        losses_te.append(loss_sup.item())


    logging.info("[Test]MSE across all predictions: {:.4e}({:.4e})".format(np.mean(losses_te), np.mean(losses_te)/num_nodes))

    logging.info("MODE:{}\t{:.4e}({:.4e})\t{:.4e}({:.4e})".format(MODE, np.min(losses_sup), np.min(losses_sup)/num_nodes, 
                                                                  np.mean(losses_te), np.mean(losses_te)/num_nodes))
    logging.info("REGION:{}".format(REGION))
    
    
def load_cfg(yaml_filepath):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    yaml_filepath : str

    Returns
    -------
    cfg : dict
    """
    # Read YAML experiment definition file
    with open(yaml_filepath, 'r') as stream:
        cfg = yaml.load(stream, Loader=yaml.FullLoader)
#     cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)
    cfg = make_paths_absolute(os.path.join(os.path.dirname(yaml_filepath), ".."), cfg)
    return cfg

def make_paths_absolute(dir_, cfg):
    """
    Make all values for keys ending with `_path` absolute to dir_.

    Parameters
    ----------
    dir_ : str
    cfg : dict

    Returns
    -------
    cfg : dict
    """
    for key in cfg.keys():
        if key.endswith("_path"):
            cfg[key] = os.path.join(dir_, cfg[key])
            cfg[key] = os.path.abspath(cfg[key])
            if not os.path.isfile(cfg[key]):
                logging.error("%s does not exist.", cfg[key])
        if type(cfg[key]) is dict:
            cfg[key] = make_paths_absolute(dir_, cfg[key])
    return cfg
    
def get_parser():
    """Get parser object."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Implementation of Differentiable Physics-informed Graph Networks',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-f", "--file",
                        dest="filename",
                        help="experiment definition file (YAML format)",
                        metavar="FILE",
                        required=True)
    parser.add_argument("--gpu",
                        type=int,
                        default=0,
                        help="gpu number: 0 or 1")
    
    parser.add_argument("--model_path",
                        dest="modelpath",
                        help="load pretrained model",
                        default=False)
    
    
    return parser

    
if __name__=="__main__":
    args = get_parser().parse_args()
    
    cfg = load_cfg(args.filename)

    torch.cuda.set_device(args.gpu)
    
    cfg['modelpath'] = args.modelpath
    
    main(cfg)