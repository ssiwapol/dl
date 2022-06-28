import torch.optim as optim

from models.base import ClassificationNetwork
from models.inception_v1 import InceptionResnetV1
from models.convnext_t import ConvNeXtT
from models.resnet34 import ResNet34, ResNet34Adj
from train import *
from test import *


if __name__=='__main__':

    ####### Classification model #######
    args = {
        "exp_name": "resnet34_04",
        "train_dir": "/home/ubuntu/data/classification/classification/train",
        "val_dir": "/home/ubuntu/data/classification/classification/dev",
        "test_dir": "/home/ubuntu/data/classification/classification/test",
        "model_dir": "/home/ubuntu/efs/model/hw2",
        "wandb_proj": "11785-hw2p2",
        "batch_size": 128,
        "lr": 0.3,
        "label_smoothing": 0.2,
        "epochs": 80,
        "kaggle_msg": "Submission 07",
        "kaggle_prj": "11-785-s22-hw2p2-classification"
    }

    # Define model
    n_class, n_train_loader = ClassificationData.find_n(args['train_dir'], args['batch_size'])
    # model = ClassificationNetwork(num_classes=n_class)
    # model = ConvNeXtT(num_classes=n_class)
    # model = InceptionResnetV1(num_classes=n_class)
    # model = ResNet34(num_classes=n_class)
    model = ResNet34Adj(num_classes=n_class)

    # Training method
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = LabelSmoothing(smoothing=args['label_smoothing'])
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(n_train_loader * args['epochs']))

    # Train model
    t = ClassificationTraining(model, optimizer, criterion, scheduler, args)
    # t = ClassificationTripleTraining(model, optimizer, criterion, scheduler, args)
    t.load_data()
    t.train(viz=True)

    # Predict from last save model
    args['exp_load'] = t.exp_name
    n_class, n_train_loader = ClassificationData.find_n(args['train_dir'], args['batch_size'])
    model = ResNet34Adj(num_classes=n_class)
    p = ClassificationPrediction(model, args)
    p.predict()


    ####### Verification model #######
    args = {
        "ver_dir": "/home/ubuntu/data/verification/verification",
        "train_dir": "/home/ubuntu/data/classification/classification/train",
        "val_dir": "/home/ubuntu/data/verification/verification/dev",
        "test_dir": "/home/ubuntu/data/verification/verification/test",
        "model_dir": "/home/ubuntu/efs/model/hw2",
        "batch_size": 64,
        "triple_margin": 0.2,
        "triple_p": 2,
        "epochs": 1000,
        "kaggle_msg": "Submission 06",
        "kaggle_prj": "11-785-s22-hw2p2-verification"
    }
    args['exp_load'] = t.exp_name

    # Define model
    n_class, n_train_loader = ClassificationData.find_n(args['train_dir'], args['batch_size'])
    model = InceptionResnetV1(num_classes=n_class)

    # Training method
    criterion = torch.nn.TripletMarginLoss(margin=args['triple_margin'], p=args['triple_p'])
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(n_train_loader * args['epochs']))

    # Train model
    t = VerificationTraining(model, optimizer, criterion, scheduler, args)
    t.load_data()
    t.load_params()
    t.train(resume=False)

    # Predict from last save model
    p = VerificationPrediction(model, args)
    p.val()
    p.predict()
