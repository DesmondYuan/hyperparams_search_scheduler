import hashlib
import torch
import argparse

POSS_VAL_NOT_LIST = 'Flag {} has an invalid list of values: {}. Length of list must be >=1'

def parse_args(args_str=None):
    parser = argparse.ArgumentParser(description='PancNet Classifier')
    # What to execute
    parser.add_argument('--train', action='store_true', default=False, help='Whether or not to train model')
    parser.add_argument('--test', action='store_true', default=False, help='Whether or not to run model on test set')
    parser.add_argument('--dev', action='store_true', default=False, help='Whether or not to run model on dev set')
    # Device level information
    parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu. Only relevant for NNs')
    parser.add_argument('--num_workers', type=int, default=8, help='num workers for each data loader [default: 4]')
    # Dataset setup
    parser.add_argument('--dataset', type=str, default='dataset_factory_A', help="Name of dataset to use.")
    parser.add_argument('--metadata_path', type=str, default='data/simu_metadata.json', help="Path of source datafile")
    parser.add_argument('--gen_index', type=int, default=0, help="Internal argument")
    # Model Hyper-params
    parser.add_argument('--model_name', type=str, default='bow', help="Model to be used.")
    parser.add_argument('--hidden_dim', type=int, default=300, help="Representation size at end of network.")
    parser.add_argument('--embedding_dim', type=int, default=128, help="Representation size at for disease code embeddings.")
    parser.add_argument('--dropout', type=float, default=0.2, help="Dropout value for the neural network model.")
    # Learning Hyper-params
    parser.add_argument('--loss_f', type=str, default="binary_cross_entropy_with_logits", help='loss function to use, available: [Xent (default), MSE]')
    parser.add_argument('--optimizer', type=str, default="adam", help='optimizer to use [default: adam]')
    parser.add_argument('--batch_splits', type=int, default=1, help='Splits batch size into smaller batches in order to fit gpu memmory limits. Optimizer step is run only after one batch size is over. Note: batch_size/batch_splits should be int [default: 1]')
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size to train neural network model.")
    parser.add_argument('--max_batches_per_train_epoch', type=int, default=10000, help='max batches to per train epoch. [default: 10000]')
    parser.add_argument('--max_batches_per_dev_epoch', type=int, default=10000, help='max batches to per dev epoch. [default: 10000]')
    parser.add_argument('--init_lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('--lr_decay', type=float, default=1., help='Decay of learning rate [default: no decay (1.)]')
    parser.add_argument('--momentum', type=float, default=0, help='Momentum to use with SGD')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 Regularization penaty [default: 0]')
    parser.add_argument('--patience', type=int, default=5, help='number of epochs without improvement on dev before halving learning rate and reloading best model [default: 5]')
    parser.add_argument('--tuning_metric', type=str, default='c_index', help='Metric to judge dev set results. Possible options include auc, loss, accuracy [default: loss]')
    parser.add_argument('--epochs', type=int, default=256, help='number of epochs for train [default: 256]')
    parser.add_argument('--linear_interpolate_risk', action='store_true', default=False, help='linearily interpolate risk from init year to actual year at cancer.') #
    parser.add_argument('--class_bal', action='store_true', default=True, help='Whether to apply a weighted sampler to balance between the classes on each batch.')
    # Where to store stuff
    parser.add_argument('--time_logger_verbose', type=int, default=2, help='Verbose of logging (1: each main, 2: each epoch, 3: each step). Default: 2.')
    parser.add_argument('--time_logger_step', type=int, default=1, help='Log the time elapse every how many iterations - 0 for no logging.')
    parser.add_argument('--save_dir', type=str, default='snapshot', help='where to dump the model')
    parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot to load[default: None]')
    parser.add_argument('--results_path', type=str, default='logs/results.p', help="Path of where to store results dict")

    if args_str is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_str.split())

    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = 'cuda' if args.cuda else 'cpu'


    # learning initial state
    args.optimizer_state = None
    args.current_epoch = None
    args.lr = None
    args.epoch_stats = None
    args.step_indx = 1

    return args


def md5(key):
    '''
    returns a hashed with md5 string of the key
    '''
    return hashlib.md5(key.encode()).hexdigest()


def parse_dispatcher_config(config):
    '''
    Parses an experiment config, and creates jobs. For flags that are expected to be a single item,
    but the config contains a list, this will return one job for each item in the list.
    :config - experiment_config

    returns: jobs - a list of flag strings, each of which encapsulates one job.
        *Example: --train --cuda --dropout=0.1 ...
    returns: experiment_axies - axies that the grid hyperparameter is hyperparametering over
    '''
    jobs = [""]
    experiment_axies = []
    hyperparameter_space = config['search_space']

    hyperparameter_space_flags = hyperparameter_space.keys()
    hyperparameter_space_flags = sorted(hyperparameter_space_flags)
    for ind, flag in enumerate(hyperparameter_space_flags):
        possible_values = hyperparameter_space[flag]
        if len(possible_values) > 1:
            experiment_axies.append(flag)

        children = []
        if len(possible_values) == 0 or type(possible_values) is not list:
            raise Exception(POSS_VAL_NOT_LIST.format(flag, possible_values))
        for value in possible_values:
            for parent_job in jobs:
                if type(value) is bool:
                    if value:
                        new_job_str = "{} --{}".format(parent_job, flag)
                    else:
                        new_job_str = parent_job
                elif type(value) is list:
                    val_list_str = " ".join([str(v) for v in value])
                    new_job_str = "{} --{} {}".format(parent_job, flag,
                                                      val_list_str)
                else:
                    new_job_str = "{} --{} {}".format(parent_job, flag, value)
                children.append(new_job_str)
        jobs = children

    return jobs, experiment_axies
