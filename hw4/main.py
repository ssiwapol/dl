import torch
import torch.nn as nn

from models.las1 import Las01
from models.las2 import Las02, Las03
from models.las3 import Las04, Las05, Las06
from models.las4 import Las07
from models.las5 import Las08, Las09
from models.las6 import Las10, Las11
from models.las7 import Las12, Las13
from models.las8 import Las15
from train import *
from test import *



if __name__=='__main__':

    ####### Model Training #######
    args = {
        "exp_name": "las_11",
        #"exp_load": "las_01",
        "train_dir": "/home/ubuntu/data/hw4p2_student_data/hw4p2_student_data/train",
        "val_dir": "/home/ubuntu/data/hw4p2_student_data/hw4p2_student_data/dev",
        "test_dir": "/home/ubuntu/data/hw4p2_student_data/hw4p2_student_data/test",
        "model_dir": "/home/ubuntu/efs/dl-hw4/model",
        "wandb_proj": "11785-hw4p2",
        "patience": 5,
        "batch_size": 64,
        "lr": 0.002,
        "epochs": 60
    }

    # Define model
    input_size, output_size = LibriTrainData.find_n(args['train_dir'])
    model = Las11(input_size, output_size)

    # Training method
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args['patience'])

    # Train model
    t = ModelTraining(model, optimizer, criterion, scheduler, args)
    t.load_data()
    t.train(viz=True)

    ####### Model Prediction #######
    args = {
        #"exp_load": "las_11",
        "train_dir": "/home/ubuntu/data/hw4p2_student_data/hw4p2_student_data/train",
        "test_dir": "/home/ubuntu/data/hw4p2_student_data/hw4p2_student_data/test",
        "model_dir": "/home/ubuntu/efs/dl-hw4/model",
        "batch_size": 64,
        "kaggle_msg": "Submission 10",
        "kaggle_prj": "11-785-s22-hw4p2"
    }
    args['exp_load'] = t.exp_name
    input_size, output_size = LibriTrainData.find_n(args['train_dir'])
    model = Las11(input_size, output_size)
    p = ModelPrediction(model, args)
    p.predict()
