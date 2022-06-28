
import os
import os.path as osp
from datetime import datetime as dt
import time
import json
import pytz
import shutil

import torch
import torchsummaryX
import wandb
from tqdm import tqdm

from dataloader import *
from utils import *


class ModelTraining():
    def __init__(self, model, optimizer, criterion, scheduler, decoder, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.decoder = decoder
        self.args = args

        # Set experiment path
        exp_suffix = dt.now(pytz.timezone('US/Eastern')).strftime("%Y%m%d-%H%M%S")
        self.exp_name = args['exp_name'] + '_' + exp_suffix
        self.exp_path = osp.join(args['model_dir'], self.exp_name)

        # Create exp_path
        if osp.isdir(self.exp_path) is False:
            os.mkdir(self.exp_path)
            print("Create dir: {}".format(self.exp_path))
        
        # Write argument variable
        with open(osp.join(self.exp_path, "args.txt"), 'w') as f:
            f.write(json.dumps(args, indent=4))
            print("Write file: {}".format(osp.join(self.exp_path, "args.txt")))

    def load_data(self):
        # Load training data
        self.train_data = LibriTrainData(self.args['train_dir'], self.args['batch_size'])
        self.train_loader = self.train_data.load(train=True)

        # Load validation data
        self.val_data = LibriTrainData(self.args['val_dir'], self.args['batch_size'])
        self.val_loader = self.val_data.load(train=False)
    
    def write_param(self):
        # Write parameters to file
        with open(osp.join(self.exp_path, "parameters.txt"), "w") as f:

            # Model architecture
            f.write('Model\n')
            f.write('Class: {}\n{}\n'.format(str(self.model.__class__), str(self.model)))
            x, y, lx, ly = iter(self.val_loader).next()
            model_sum = str(torchsummaryX.summary(self.model, x.to(self.device), lx))
            f.write(model_sum)
            l = '-' * 70
            f.write('\n\n{}\n'.format(l))

            # Criterion (Loss function)
            f.write('Criterion\n')
            criterion_param = json.dumps({k: str(v) for k, v in self.criterion.__dict__.items()}, indent=4)
            f.write('Class: {}\n{}\n{}\n'.format(self.criterion.__class__, criterion_param, l))

            # Optimizer
            f.write('Optimizer\nClass: {}\n{}\n{}\n'.format(self.optimizer.__class__, str(self.optimizer), l))

            # Scheduler
            f.write('Scheduler\n')
            scheduler_param = json.dumps({k: str(v) for k, v in self.scheduler.__dict__.items()}, indent=4)
            f.write('Class: {}\n{}\n{}\n'.format(self.scheduler.__class__, scheduler_param, l))

            # Decoder
            f.write('Decoder\n')
            decoder_param = json.dumps({k: str(v) for k, v in self.decoder.__dict__.items()}, indent=4)
            f.write('Class: {}\n{}\n{}\n'.format(self.decoder.__class__, decoder_param, l))
            print("Write file: {}".format(osp.join(self.exp_path, "parameters.txt")))


    def save_checkpoint(self, metrics, best_model):
        # Save model
        exp_run_path = osp.join(self.exp_path, 'run.tar')
        checkpoint = metrics
        checkpoint['model_state_dict'] = self.model.state_dict()
        checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(checkpoint, exp_run_path)
        print("Save model: {}".format(exp_run_path))

        # if this is the best model save model and metrics to model.tar
        if best_model:
            exp_best_path = os.path.join(self.exp_path, 'model.tar')
            torch.save(checkpoint, exp_best_path)
            print("Save model: {}".format(exp_best_path))

    def setup_wandb(self, resume):
        if resume:
            with open(osp.join(self.args['model_dir'], self.args['exp_load'], "wandb.txt")) as f:
                wandb_id = f.readline()
        else:
            wandb_id = wandb.util.generate_id()
        with open(osp.join(self.exp_path, "wandb.txt"), "w") as f:
            f.write(str(wandb_id))
            
        
        wandb_conf = {
            'model': str(self.model.__class__.__name__),
            'criterion': str(self.criterion.__class__.__name__),
            'optimizer': str(self.optimizer.__class__.__name__),
            'scheduler': str(self.scheduler.__class__.__name__),
            'lr': self.args['lr']
        }

        wandb.init(project=self.args['wandb_proj'], name=self.args['exp_name'], 
                   id=wandb_id, resume='allow', config=wandb_conf)
        

    def train(self, viz=True):
        self.write_param()
        print('Using device:', next(self.model.parameters()).device)
        epochs = self.args['epochs']
        resume = True if self.args.get('exp_load') else False

        # Setup wandb
        if viz:
            self.setup_wandb(resume)

        # Using Mixed precision training
        scaler = torch.cuda.amp.GradScaler()

        # Load check if resume
        if resume:
            checkpoint = torch.load(osp.join(self.args['model_dir'], self.args['exp_load'], 'run.tar'))
            epoch = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            shutil.copyfile(
                osp.join(self.args['model_dir'], self.args['exp_load'], 'metrics.txt'), 
                osp.join(self.exp_path, 'metrics.txt'))
            total_time = checkpoint['total_time']
        else:
            epoch = 0
            total_time = 0
            
        best_dist = 0
        while epoch < epochs:
            start_time = time.time()

            # Training
            self.model.train()

            prog_bar = tqdm(total=len(self.train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')
            
            # Clear cache
            torch.cuda.empty_cache()
            trn_loss = 0

            for i, (x, y, lx, ly) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                x = x.to(self.device)
                y = y.to(self.device)

                # For FP16
                with torch.cuda.amp.autocast():     
                    h, lh = self.model(x, lx)
                    loss = self.criterion(torch.transpose(h, 0, 1), y, lh, ly)

                # Update # correct & loss
                trn_loss += float(loss)

                # Another things need for FP16. 
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                # # Step scheduler
                # self.scheduler.step()

                # Update tqdm
                prog_bar.set_postfix(loss='{:.04f}'.format(trn_loss/(i+1)))
                prog_bar.update()
            
            # Close tqdm
            prog_bar.close()

            # Validation
            prog_bar = tqdm(total=len(self.val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')

            self.model.eval()

            val_dist = 0
            for i, (x, y, lx, ly) in enumerate(self.val_loader):
                
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.no_grad():
                    h, lh = self.model(x, lx)
                    loss = self.criterion(torch.transpose(h, 0, 1), y, lh, ly)
                
                
                # Update dist
                dist = cal_levenshtein(h, y, lh, ly, self.decoder)
                val_dist += dist

                # Update tqdm
                prog_bar.set_postfix(dist='{:.04f}'.format(val_dist/(i+1)))
                prog_bar.update()
                
            prog_bar.close()
            
            # Calculate metrics
            trn_loss = trn_loss / len(self.train_loader)
            val_dist = val_dist / len(self.val_loader)
            print("Epoch {}/{}: Train Loss {:.04f}, Val Levenshtein Distance {:.04f}, Lr {:.08}".format(
                epoch + 1, epochs, 
                trn_loss, val_dist, 
                self.optimizer.param_groups[0]['lr']
            ))

            # Step scheduler
            self.scheduler.step(val_dist)

            total_time = (time.time() - start_time) + total_time
            metrics = {
                'epoch': epoch,
                'train_loss': trn_loss,
                'val_dist': val_dist,
                'lr': self.optimizer.param_groups[0]['lr'],
                'total_time': total_time
            }
            # Write metrics
            with open(osp.join(self.exp_path, "metrics.txt"), 'a') as f:
                f.write(str(metrics)+'\n')
            # Write log to wandb
            if viz:
                wandb.log({'val_dist': val_dist, 'train_loss': trn_loss})
            
            # Test best model
            if epoch == 0 or resume:
                best_model = True
                best_dist = val_dist
                resume = False
            elif val_dist < best_dist:
                best_model = True
                best_dist = val_dist
            else:
                best_model = False
            # Save checkpoint
            self.save_checkpoint(metrics, best_model)

            epoch += 1
