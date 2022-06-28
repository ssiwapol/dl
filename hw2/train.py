
import os
import os.path as osp
from datetime import datetime as dt
import time
import json
import pytz
import shutil

import torch
from torchinfo import summary
from sklearn.metrics import roc_auc_score
import wandb
from tqdm import tqdm

from dataloader import *
from utils import *


class ClassificationTraining():
    def __init__(self, model, optimizer, criterion, scheduler, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
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
        self.train_data = ClassificationData(self.args['train_dir'], self.args['batch_size'])
        self.train_loader = self.train_data.load(train=True, augment=True)

        # Load validation data
        self.val_data = ClassificationData(self.args['val_dir'], self.args['batch_size'])
        self.val_loader = self.val_data.load(train=False)
    
    def write_param(self):
        # Write parameters to file
        with open(osp.join(self.exp_path, "parameters.txt"), "w") as f:

            # Model architecture
            f.write('Model\n')
            f.write('Class: {}\n'.format(str(self.model.__class__)))
            model_sum = str(summary(self.model, input_size=self.train_data.input_size))
            f.write(model_sum)
            l = '-' * 90
            f.write('\n\n{}\n'.format(l))

            # Criterion (Loss function)
            f.write('Criterion\n')
            criterion_param = json.dumps({k: str(v) for k, v in self.criterion.__dict__.items()}, indent=4)
            f.write('Class: {}\n{}\n{}\n'.format(self.criterion.__class__, criterion_param, l))

            # Optimizer
            f.write('Optimizer\nClass: {}\n{}\n{}\n'.format(self.optimizer.__class__, str(self.optimizer), l))
            f.write('Scheduler\n')

            # Scheduler
            scheduler_param = json.dumps({k: str(v) for k, v in self.scheduler.__dict__.items()}, indent=4)
            f.write('Class: {}\n{}\n{}\n'.format(self.scheduler.__class__, scheduler_param, l))
            print("Write file: {}".format(osp.join(self.exp_path, "parameters.txt")))
        
        n = cal_num_params(self.model)
        print("Number of Params: {:,}".format(n))

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
        batch_size = self.args['batch_size']
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
            
        best_loss = 0
        while epoch < epochs:
            start_time = time.time()

            # Training
            self.model.train()
            batch_bar = tqdm(total=len(self.train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

            trn_n_correct = 0
            trn_loss = 0

            for i, (x, y, _) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                x = x.to(self.device)
                y = y.to(self.device)

                # For FP16
                with torch.cuda.amp.autocast():     
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)

                # Update # correct & loss
                trn_n_correct += int((torch.argmax(outputs, axis=1) == y).sum())
                trn_loss += float(loss)

                # Another things need for FP16. 
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                # Step scheduler
                self.scheduler.step()

                # Update tqdm
                batch_bar.set_postfix(
                    acc="{:.04f}%".format(100 * trn_n_correct / ((i + 1) * batch_size)),
                    loss="{:.04f}".format(float(trn_loss / (i + 1))))
                batch_bar.update()
            
            # Close tqdm
            batch_bar.close()

            # Validation
            batch_bar = tqdm(total=len(self.val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')

            self.model.eval()

            val_n_correct = 0
            val_loss = 0

            for i, (x, y, _) in enumerate(self.val_loader):
                
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.no_grad():
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)
                
                # Update # correct & loss
                val_n_correct += int((torch.argmax(outputs, axis=1) == y).sum())
                val_loss += float(loss)

                # Update tqdm
                batch_bar.set_postfix(
                    acc="{:.04f}%".format(100 * val_n_correct / ((i + 1) * batch_size)),
                    loss="{:.04f}".format(float(val_loss / (i + 1))))
                batch_bar.update()
                
            batch_bar.close()
            
            # Calculate metrics
            trn_acc = 100 * trn_n_correct / (len(self.train_loader) * batch_size)
            trn_loss = float(trn_loss / len(self.train_loader))
            val_acc = 100 * val_n_correct / (len(self.val_loader) * batch_size)
            val_loss = float(val_loss / len(self.val_loader))
            print("Epoch {}/{}: Train Acc {:.04f}%, Train Loss {:.04f}, Val Acc {:.04f}%, Val Loss {:.04f}".format(
                epoch + 1, epochs,
                trn_acc, trn_loss,
                val_acc, val_loss)
            )
            total_time = (time.time() - start_time) + total_time
            metrics = {
                'epoch': epoch,
                'train_acc': trn_acc,
                'train_loss': trn_loss,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'total_time': total_time
            }
            # Write metrics
            with open(osp.join(self.exp_path, "metrics.txt"), 'a') as f:
                f.write(str(metrics)+'\n')
            # Write log to wandb
            if viz:
                wandb.log({'val_loss': val_loss, 'val_acc': val_acc,
                           'train_loss': trn_loss,'train_acc': trn_acc})
            
            # Test best model
            if epoch == 0 or resume:
                best_model = True
                best_loss = val_loss
                resume = False
            elif val_loss < best_loss:
                best_model = True
                best_loss = val_loss
            else:
                best_model = False
            # Save checkpoint
            self.save_checkpoint(metrics, best_model)

            epoch += 1


class ClassificationTripleTraining(ClassificationTraining):
    def load_data(self):
        self.train_data = TripleData(self.args['train_dir'], self.args['batch_size'])
        self.train_loader = self.train_data.load(train=True, augment=True)

        self.val_data = ClassificationData(self.args['val_dir'], self.args['batch_size'])
        self.val_loader = self.val_data.load(train=False)

    def train(self, viz=True):
        self.write_param()
        print('Using device:', next(self.model.parameters()).device)
        batch_size = self.args['batch_size']
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
            total_time = checkpoint['total_time']
        else:
            epoch = 0
            total_time = 0
        
        best_loss = 0
        criterion_t = torch.nn.TripletMarginLoss(margin=self.args['triple_margin'], p=self.args['triple_p'])
        while epoch < epochs:
            start_time = time.time()

            # Training
            self.model.train()
            batch_bar = tqdm(total=len(self.train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

            trn_n_correct = 0
            trn_loss = 0

            for i, (a, p, n) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                x = a[0].to(self.device)
                y = a[1].to(self.device)
                pos = p[0].to(self.device)
                neg = n[0].to(self.device)

                # For FP16
                with torch.cuda.amp.autocast():     
                    anchor = self.model(x, return_feats=True)
                    pos = self.model(pos, return_feats=True)
                    neg = self.model(neg, return_feats=True)
                    outputs = self.model(x)
                    l1 = self.criterion(outputs, y) / 100
                    l2 = criterion_t(anchor, pos, neg)
                    loss = l1 + l2
                
                # Update # correct & loss
                trn_n_correct += int((torch.argmax(outputs, axis=1) == y).sum())
                trn_loss += float(loss)

                # Another things need for FP16. 
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                # Step scheduler
                self.scheduler.step()

                # Update tqdm
                batch_bar.set_postfix(
                    acc="{:.04f}%".format(100 * trn_n_correct / ((i + 1) * batch_size)),
                    loss="{:.04f}".format(float(trn_loss / (i + 1))))
                batch_bar.update()
            
            # Close tqdm
            batch_bar.close()

            # Validation
            batch_bar = tqdm(total=len(self.val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')

            self.model.eval()

            val_n_correct = 0
            val_loss = 0

            for i, (x, y, _) in enumerate(self.val_loader):
                
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.no_grad():
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)
                
                # Update # correct & loss
                val_n_correct += int((torch.argmax(outputs, axis=1) == y).sum())
                val_loss += float(loss)

                # Update tqdm
                batch_bar.set_postfix(
                    acc="{:.04f}%".format(100 * val_n_correct / ((i + 1) * batch_size)),
                    loss="{:.04f}".format(float(val_loss / (i + 1))))
                batch_bar.update()
                
            batch_bar.close()
            
            # Calculate metrics
            trn_acc = 100 * trn_n_correct / (len(self.train_loader) * batch_size)
            trn_loss = float(trn_loss / len(self.train_loader))
            val_acc = 100 * val_n_correct / (len(self.val_loader) * batch_size)
            val_loss = float(val_loss / len(self.val_loader))
            print("Epoch {}/{}: Train Acc {:.04f}%, Train Loss {:.04f}, Val Acc {:.04f}%, Val Loss {:.04f}".format(
                epoch + 1, epochs,
                trn_acc, trn_loss,
                val_acc, val_loss)
            )
            total_time = (time.time() - start_time) + total_time
            metrics = {
                'epoch': epoch,
                'train_acc': trn_acc,
                'train_loss': trn_loss,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'total_time': total_time
            }
            # Write metrics
            with open(osp.join(self.exp_path, "metrics.txt"), 'a') as f:
                f.write(str(metrics)+'\n')
            # Write log to wandb
            if viz:
                wandb.log({'val_loss': val_loss, 'val_acc': val_acc,
                           'train_loss': trn_loss,'train_acc': trn_acc})
            
            # Test best model
            if epoch == 0 or resume:
                best_model = True
                best_loss = val_loss
                resume = False
            elif val_loss < best_loss:
                best_model = True
                best_loss = val_loss
            else:
                best_model = False
            # Save checkpoint
            self.save_checkpoint(metrics, best_model)

            epoch += 1


class VerificationTraining():
    def __init__(self, model, optimizer, criterion, scheduler, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.args = args
        
        # Write argument variable
        self.exp_path = osp.join(self.args['model_dir'], self.args['exp_load'])
        with open(osp.join(self.exp_path, "args_v.txt"), 'w') as f:
            f.write(json.dumps(args, indent=4))
            print("Write file: {}".format(osp.join(self.exp_path, "args_v.txt")))

    def load_data(self):
        # Load training data
        self.train_data = TripleData(self.args['train_dir'], self.args['batch_size'])
        self.train_loader = self.train_data.load(train=True)

        # Load validation data
        self.val_data = VerificationData(self.args['val_dir'], self.args['batch_size'])
        self.val_loader = self.val_data.load()
    
    def load_params(self):
        # Load model weight
        checkpoint = torch.load(os.path.join(self.exp_path, 'model.tar'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        #last_lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
    
    def write_param(self):
        # Write parameters to file
        with open(osp.join(self.exp_path, "parameters_v.txt"), "w") as f:

            # Model architecture
            f.write('Model\n')
            f.write('Class: {}\n'.format(str(self.model.__class__)))
            model_sum = str(summary(self.model, input_size=self.train_data.input_size))
            f.write(model_sum)
            l = '-' * 90
            f.write('\n\n{}\n'.format(l))

            # Criterion (Loss function)
            f.write('Criterion\n')
            criterion_param = json.dumps({k: str(v) for k, v in self.criterion.__dict__.items()}, indent=4)
            f.write('Class: {}\n{}\n{}\n'.format(self.criterion.__class__, criterion_param, l))

            # Optimizer
            f.write('Optimizer\nClass: {}\n{}\n{}\n'.format(self.optimizer.__class__, str(self.optimizer), l))
            f.write('Scheduler\n')

            # Scheduler
            scheduler_param = json.dumps({k: str(v) for k, v in self.scheduler.__dict__.items()}, indent=4)
            f.write('Class: {}\n{}\n{}\n'.format(self.scheduler.__class__, scheduler_param, l))
            print("Write file: {}".format(osp.join(self.exp_path, "parameters_v.txt")))
    
    def save_checkpoint(self, metrics, best_model):
        # Save model
        exp_run_path = osp.join(self.exp_path, 'run_v.tar')
        checkpoint = metrics
        checkpoint['model_state_dict'] = self.model.state_dict()
        checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(checkpoint, exp_run_path)
        print("Save model: {}".format(exp_run_path))

        # if this is the best model save model and metrics to model.tar
        if best_model:
            exp_best_path = os.path.join(self.exp_path, 'model_v.tar')
            torch.save(checkpoint, exp_best_path)
            print("Save model: {}".format(exp_best_path))

    def val_auc(self):
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
        return auc

    def train(self, resume=False):
        self.write_param()
        print('Using device:', next(self.model.parameters()).device)
        batch_size = self.args['batch_size']
        epochs = self.args['epochs']
        
        auc = self.val_auc()
        print('Current AUC = {:.04f}'.format(auc))

        # Using Mixed precision training
        scaler = torch.cuda.amp.GradScaler()

        # Load check if resume
        if resume:
            checkpoint = torch.load(osp.join(self.args['model_dir'], self.args['exp_load'], 'run_v.tar'))
            epoch = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            total_time = checkpoint['total_time']
        else:
            if os.path.exists(osp.join(self.exp_path, "metrics_v.txt")):
                os.remove(osp.join(self.exp_path, "metrics_v.txt"))
            epoch = 0
            total_time = 0
        
        best_auc = 0
        while epoch < epochs:
            start_time = time.time()

            # Training
            self.model.train()
            batch_bar = tqdm(total=len(self.train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

            trn_loss = 0

            for i, (a, p, n) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                anchor = a[0].to(self.device)
                pos = p[0].to(self.device)
                neg = n[0].to(self.device)

                # For FP16
                with torch.cuda.amp.autocast():     
                    anchor = self.model(anchor, return_feats=True)
                    pos = self.model(pos, return_feats=True)
                    neg = self.model(neg, return_feats=True)
                    loss = self.criterion(anchor, pos, neg)

                # Update # correct & loss
                trn_loss += float(loss)

                # Another things need for FP16. 
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                # Step scheduler
                self.scheduler.step()

                # Update tqdm
                batch_bar.set_postfix(loss="{:.04f}".format(float(trn_loss / (i + 1))))
                batch_bar.update()
            
            # Close tqdm
            batch_bar.close()

            # Validation
            auc = self.val_auc()
            
            # Calculate metrics
            trn_loss = float(trn_loss / len(self.train_loader))
            val_auc = auc
            print("Epoch {}/{}: Train Loss {:.04f}, Val AUC {:.04f}".format(
                epoch + 1, epochs,
                trn_loss, val_auc)
            )
            total_time = (time.time() - start_time) + total_time
            metrics = {
                'epoch': epoch,
                'train_loss': trn_loss,
                'val_auc': val_auc,
                'total_time': total_time
            }
            # Write metrics
            with open(osp.join(self.exp_path, "metrics_v.txt"), 'a') as f:
                f.write(str(metrics)+'\n')
            
            # Test best model
            if epoch == 0 or resume:
                best_model = True
                best_auc = val_auc
                resume = False
            elif val_auc > best_auc:
                best_model = True
                best_auc = val_auc
            else:
                best_model = False
            # Save checkpoint
            self.save_checkpoint(metrics, best_model)

            epoch += 1
