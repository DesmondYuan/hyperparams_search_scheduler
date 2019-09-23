import json
from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(realpath(__file__))))
import numpy as np
import pdb
import pickle
from utils.parsing import parse_args
from utils.time_logger import time_logger
from collections import Counter


if __name__ == '__main__':
    args = parse_args()
    save_path = args.results_path
    logger_main = time_logger(1, hierachy = 5) if args.time_logger_verbose>=1 else time_logger(0)
    logger_main.log("Now main.py starts...")
    print("CUDA:", torch.cuda.is_available())

    ##########################################################
    print("Loading Dataset...\n")
    '''
    train_data, dev_data, test_data, args = dataset_factory.get_dataset(args)
    '''
    logger_main.log("Load datasets")

    ##########################################################
    print("Building model...\n")
    '''
    model = model_factory.get_model(args)
    print(model)
    '''
    logger_main.log("Build model")

    ##########################################################
    print("Session starts...\n")
    if args.train:
        print("Training model...\n")
        '''
        epoch_stats, model = train.train_model(train_data, dev_data, model, args)
        print("Save test results to {}".format(save_path))
        args_dict = vars(args)
        pickle.dump(args_dict, open(save_path, 'wb'))
        '''
        logger_main.log("TRAINING")

    if args.dev:
        print("Evaluating model on dev...\n")
        '''
        args.dev_stats = train.eval_model(dev_data, 'dev', model, args)
        print("Save test results to {}".format(save_path))
        args_dict = vars(args)
        pickle.dump(args_dict, open(save_path, 'wb'))
        '''
        logger_main.log("VALIDATION")

    if args.test:
        print("Evaluating model on test...\n")
        '''
        args.test_stats = train.eval_model(test_data, 'test', model, args)
        print("Save test results to {}".format(save_path))
        args_dict = vars(args)
        pickle.dump(args_dict, open(save_path, 'wb'))
        '''
        logger_main.log("TESTING")

    logger_main.log("Main.py finishes.")
