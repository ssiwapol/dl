import os.path as osp
from datetime import datetime as dt
import pytz

import torch
from tqdm import tqdm

from dataloader import *
from utils import *


class ModelPrediction:
    def __init__(self, model, decoder, args):
        self.args = args

        # Load model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.decoder = decoder
        
        # Load model weight
        checkpoint = torch.load(osp.join(args['model_dir'], args['exp_load'], 'model.tar'))
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Print model detail
        print('Using device:', next(model.parameters()).device)
        print('Predict from model: {} Train Loss={:.6f}, Val Dist={:.2f}%, epoch={})'.format(
            args['exp_load'], 
            checkpoint['train_loss'], checkpoint['val_dist'], 
            checkpoint['epoch']
            ))
        
        # Load test dataset / dataloader
        self.test_data = LibriTestData(args['test_dir'], args['batch_size'])
        self.test_loader = self.test_data.load()
        
    def predict(self):
        # Prediction
        self.model.eval()
        batch_bar = tqdm(total=len(self.test_loader), dynamic_ncols=True, leave=False, position=0, desc='Pred')

        self.y_pred = []
        for i, (x, lx) in enumerate(self.test_loader):
            
            # Predict to list
            x = x.to(self.device)
            with torch.no_grad():
                h, lh = self.model(x, lx)
            y_decode = decode_h(h, lh, self.decoder)
            self.y_pred.extend(y_decode)
            
            batch_bar.update()

        batch_bar.close()

        # Write file
        suffix = dt.now(pytz.timezone('US/Eastern')).strftime("%Y%m%d-%H%M%S")
        result_path = osp.join(self.args['model_dir'], self.args['exp_load'], 'submission_{}.csv'.format(suffix))
        with open(result_path, "w") as f:
            f.write("id,predictions\n")
            for i in range(len(self.y_pred)):
                f.write("{},{}\n".format(i, self.y_pred[i]))
        print("Write file: {}".format(result_path))

        # Submit to kaggle
        if self.args.get('kaggle_msg'):
            k = 'kaggle competitions submit -c {} -f {} -m "{}"'.format(
                self.args['kaggle_prj'], result_path, self.args['kaggle_msg'])
            stream = os.popen(k)
            output = stream.read()
            print(output)
