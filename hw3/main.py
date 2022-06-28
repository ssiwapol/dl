import torch
import torch.nn as nn
from ctcdecode import CTCBeamDecoder

from models.base import Network01, Network02, Network03, Network04
from models.resnet1 import EmbResNet34_01, EmbResNet34_02
from models.resnet import EmbResNet34_03, EmbResNet34_04, EmbResNet34_05 
from models.resnet import EmbResNet50_01, EmbResNet50_02, EmbResNet50_03
from train import *
from test import *
from phonemes import PHONEMES



if __name__=='__main__':

    ####### Model Training #######
    args = {
        "exp_name": "resnet34_04f",
        #"exp_load": "resnet34_04f_20220402-004248",
        "train_dir": "/home/ubuntu/data/hw3p2_student_data/hw3p2_student_data/train",
        "val_dir": "/home/ubuntu/data/hw3p2_student_data/hw3p2_student_data/dev",
        "test_dir": "/home/ubuntu/data/hw3p2_student_data/hw3p2_student_data/test",
        "model_dir": "/home/ubuntu/efs/dl-hw3/model",
        "wandb_proj": "11785-hw2p3",
        "beam_width": 5,
        "patience": 5,
        "batch_size": 64,
        "lr": 0.002,
        "epochs": 100
    }

    # Define model
    input_size, output_size = LibriTrainData.find_n(args['train_dir'])
    model = EmbResNet34_04(input_size, output_size)

    # Training method
    criterion = nn.CTCLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args['patience'])
    decoder = CTCBeamDecoder(PHONEMES, log_probs_input=True, beam_width=args['beam_width'])

    # Train model
    t = ModelTraining(model, optimizer, criterion, scheduler, decoder, args)
    t.load_data()
    t.train(viz=True)

    ####### Model Prediction #######
    args = {
        #"exp_load": "resnet34_04_20220331-101337",
        "train_dir": "/home/ubuntu/data/hw3p2_student_data/hw3p2_student_data/train",
        "test_dir": "/home/ubuntu/data/hw3p2_student_data/hw3p2_student_data/test",
        "model_dir": "/home/ubuntu/efs/dl-hw3/model",
        "batch_size": 64,
        "beam_width": 50,
        "kaggle_msg": "Submission 10",
        "kaggle_prj": "11-785-s22-hw3p2"
    }
    args['exp_load'] = t.exp_name
    input_size, output_size = LibriTrainData.find_n(args['train_dir'])
    model = EmbResNet34_04(input_size, output_size)
    decoder = CTCBeamDecoder(PHONEMES, log_probs_input=True, beam_width=args['beam_width'])
    p = ModelPrediction(model, decoder, args)
    p.predict()
