import os
import os.path as osp

import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from dataloader import *
from utils import *
from models.base import ClassificationNetwork
from models.inception_v1 import InceptionResnetV1
from models.convnext_t import ConvNeXtT


class ClassificationPrediction:
    def __init__(self, model, args):
        self.args = args

        # Load model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        # Load model weight
        checkpoint = torch.load(os.path.join(args['model_dir'], args['exp_load'], 'model.tar'))
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Print model detail
        print('Using device:', next(model.parameters()).device)
        print('Predict from model: {} (Train Acc={:.2f}%, Train Loss={:.6f}, Val Acc={:.2f}%, Val Loss={:.2f}%, epoch={})'.format(
            args['exp_load'], 
            checkpoint['train_acc'], checkpoint['train_loss'],
            checkpoint['val_acc'], checkpoint['val_loss'], 
            checkpoint['epoch']
            ))
        
        # Load test dataset / dataloader
        self.test_data = ClassificationData(args['test_dir'], args['batch_size'])
        self.test_loader = self.test_data.load(train=False)
        
    def predict(self):
        # Prediction
        self.model.eval()
        batch_bar = tqdm(total=len(self.test_loader), dynamic_ncols=True, leave=False, position=0, desc='Test')

        self.x_file = []
        self.y_pred = []
        for i, (x, _, p) in enumerate(self.test_loader):

            # Get file name from file path
            self.x_file.extend([osp.basename(j) for j in p])

            # Predict to list
            x_pred = x.to(self.device)
            with torch.no_grad():
                outputs = self.model(x_pred)
            self.y_pred.extend(torch.argmax(outputs, axis=1).tolist())

            batch_bar.update()

        batch_bar.close()

        # Write file
        result_path = osp.join(self.args['model_dir'], self.args['exp_load'], 'classification_submission.csv')
        with open(result_path, "w") as f:
            f.write("id,label\n")
            for i in range(len(self.x_file)):
                f.write("{},{}\n".format(self.x_file[i], self.y_pred[i]))
            print("Write file: {}".format(result_path))
        
        # Submit to kaggle
        if self.args.get('kaggle_msg'):
            k = 'kaggle competitions submit -c {} -f {} -m "{}"'.format(
                self.args['kaggle_prj'], result_path, self.args['kaggle_msg'])
            stream = os.popen(k)
            output = stream.read()
            print(output)


class VerificationPrediction:
    def __init__(self, model, args):
        self.args = args

        # Load model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # Load model weight
        if osp.exists(osp.join(os.path.join(args['model_dir'], args['exp_load'], 'model_v.tar'))):
            checkpoint = torch.load(os.path.join(args['model_dir'], args['exp_load'], 'model_v.tar'))
        else:
            checkpoint = torch.load(os.path.join(args['model_dir'], args['exp_load'], 'model.tar'))
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load data
        self.val_data = VerificationData(args['val_dir'], args['batch_size'])
        self.val_loader = self.val_data.load()
        self.test_data = VerificationData(args['test_dir'], args['batch_size'])
        self.test_loader = self.test_data.load()

    def val(self):
        feats_dict = find_feats(self.val_loader, self.model, self.device)

        # Loop through CSV to find similarity
        val_veri_csv = osp.join(self.args['ver_dir'], "verification_dev.csv")
        pred_similarities = []
        gt_similarities = []
        for line in tqdm(open(val_veri_csv).read().splitlines()[1:], 
                        position=0, leave=False, desc='Similarity'):
            img_path1, img_path2, gt = line.split(",")

            # Calculate cosine similarity
            similarity = cal_similarity(img_path1, img_path2, feats_dict)
            pred_similarities.append(similarity)

            gt_similarities.append(int(gt))

        # Transform to array
        pred_similarities = np.array(pred_similarities)
        gt_similarities = np.array(gt_similarities)

        # Calculate AUC-ROC
        auc = roc_auc_score(gt_similarities, pred_similarities)
        print('Predict from model: {} (Val AUC={:.6f})'.format(self.args['exp_load'], auc))
    
    def predict(self):
        feats_dict = find_feats(self.test_loader, self.model, self.device, prefix='test/')

        # Loop through CSV to find similarity
        pred_similarities = []
        test_veri_csv = osp.join(self.args['ver_dir'], 'verification_test.csv')
        for line in tqdm(open(test_veri_csv).read().splitlines()[1:], 
                        position=0, leave=False, desc='Similarity'):
            img_path1, img_path2 = line.split(",")

            # Calculate cosine similarity
            similarity = cal_similarity(img_path1, img_path2, feats_dict)
            pred_similarities.append(similarity)

        # Transform to array
        pred_similarities = np.array(pred_similarities)

        # Write file
        result_path = osp.join(self.args['model_dir'], self.args['exp_load'], 'verification_submission.csv')
        with open(result_path, "w") as f:
            f.write("id,match\n")
            for i in range(len(pred_similarities)):
                f.write("{},{}\n".format(i, pred_similarities[i]))
            print("Write file: {}".format(result_path))

        # Submit to kaggle
        if self.args.get('kaggle_msg'):
            k = 'kaggle competitions submit -c {} -f {} -m "{}"'.format(
                self.args['kaggle_prj'], result_path, self.args['kaggle_msg'])
            stream = os.popen(k)
            output = stream.read()
            print(output)


if __name__=='__main__':
    # AWS
    args = {
        "exp_load": "inception_01_20220312-224230",
        "train_dir": "/home/ubuntu/data/classification/classification/train",
        "test_dir": "/home/ubuntu/data/classification/classification/test",
        "model_dir": "/home/ubuntu/efs/model/hw2",
        "batch_size": 128,
        "kaggle_msg": "Submission 04",
        "kaggle_prj": "11-785-s22-hw2p2-classification"
    }

    # Colab
    # args = {
    #     "exp_load": "inception_01_20220312-224230",
    #     "train_dir": "/content/classification/classification/train",
    #     "test_dir": "/content/classification/classification/test",
    #     "model_dir": "/content/drive/MyDrive/CMU/dev/dl-hw2/model",
    #     "batch_size": 128,
    #     "kaggle_msg": "Submission 04",
    #     "kaggle_prj": "11-785-s22-hw2p2-classification"
    # }

    n_class, n_train_loader = ClassificationData.find_n(args['train_dir'], args['batch_size'])
    model = InceptionResnetV1(num_classes=n_class)
    p = ClassificationPrediction(model, args)
    p.predict()
