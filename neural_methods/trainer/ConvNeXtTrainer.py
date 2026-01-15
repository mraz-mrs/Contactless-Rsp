import torch
import torch.nn as nn
from tqdm import tqdm
from neural_methods.utils.select_model import get_model
import os
import matplotlib.pyplot as plt
import time
from timm.models.layers import trunc_normal_
import pickle
from neural_methods.loss.pearson import PearsonLoss
import json

class ConvNeXtTrainer:
    def __init__(self, configs):
        self.configs = configs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = configs.MODEL.NAME
        self.model = get_model(configs)

        if configs.TRAIN.CONTINUE_TRAINING and configs.MODEL.MODEL_PATH:
            self.model.load_state_dict(torch.load(configs.MODEL.MODEL_PATH))
            print(f"Model {self.model_name} loaded from {configs.MODEL.MODEL_PATH}")


        self.batch_size = configs.TRAIN.BATCH_SIZE
        self.lr = configs.TRAIN.LR
        self.weight_decay = configs.TRAIN.WEIGHT_DECAY
        self.epochs = configs.TRAIN.EPOCHS
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.configs.TRAIN.SCHEDULER == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.configs.TRAIN.EPOCHS)    
        else:
            self.scheduler = None

        self.criterion = torch.nn.MSELoss()
        # self.criterion = PearsonLoss()
        
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        self.min_val_loss = float('inf')
        
        self.train_loss_ls = []
        self.val_loss_ls = []
        self.timestamp = time.strftime("%Y%m%d%H%M", time.localtime())
        self.checkpoint_path = os.path.join(configs.MODEL.SAVEPATH, f'{self.model_name}_{self.timestamp}', 'checkpoints')
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        self.figs_path = os.path.join(configs.MODEL.SAVEPATH, f'{self.model_name}_{self.timestamp}', 'figs')
        if not os.path.exists(self.figs_path):
            os.makedirs(self.figs_path)
        self.best_checkpoint_path = os.path.join(configs.MODEL.SAVEPATH, f'{self.model_name}_{self.timestamp}', 'best_checkpoints')
        if not os.path.exists(self.best_checkpoint_path):
            os.makedirs(self.best_checkpoint_path)
        
        
    def _load_state_dict(self, model_path):
        checkpoint = torch.load(model_path, map_location='cpu')

        checkpoint_model = checkpoint
        state_dict = self.model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        _model = get_model(self.configs)
        _model.load_state_dict(checkpoint_model, strict=False)
        _model.to(self.device)
        return _model
        
    def save_model(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, f'{self.model_name}_{epoch}.pth'))
        print(f"Model {self.model_name} saved at epoch {epoch}")

    def save_best_model(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.best_checkpoint_path, f'{self.model_name}_best.pth'))
        print(f"Best model {self.model_name} saved at epoch {epoch}")

    def save_loss_plot(self):
        # 清除之前的图
        plt.clf()
        plt.plot(self.train_loss_ls, label='train')
        plt.plot(self.val_loss_ls, label='val')
        plt.legend()
        plt.savefig(os.path.join(self.figs_path, f'{self.model_name}_loss.png'))
    
    def train(self, train_loader, val_loader):
        for epoch in range(self.epochs):
            import time
            start_time = time.time()
            print(f"\n\nTraining at Epoch {epoch}")
            train_epoch_loss = []
            self.model.train()
            tbar = tqdm(train_loader, ncols=80)
            for idx, (stmap, rsp_label) in enumerate(tbar):
                tbar.set_description(f"Epoch {epoch+1}/{self.epochs}")
                stmap = stmap.to(self.device)
                rsp_label = rsp_label.to(self.device)
                
                self.optimizer.zero_grad()
                rsp_pred = self.model(stmap)
                train_loss = self.criterion(rsp_pred, rsp_label)
                train_loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                tbar.set_postfix(loss=train_loss.item())
                train_epoch_loss.append(train_loss.item())
            self.validate(val_loader)
            print(f"Train Loss: {sum(train_epoch_loss) / len(train_epoch_loss)}")
            print(f"Time Elapsed: {time.time() - start_time}")
            self.train_loss_ls.append(sum(train_epoch_loss) / len(train_epoch_loss))
            if epoch % self.configs.MODEL.SAVE_INTERVAL == 0:
                self.save_model(epoch)

            if self.val_loss_ls[-1] < self.min_val_loss:
                self.min_val_loss = self.val_loss_ls[-1]
                self.save_best_model(epoch)

            self.save_loss_plot()
        self.inference(os.path.join(self.best_checkpoint_path, f'{self.model_name}_best.pth'), train_loader, dataset='train')
        self.inference(os.path.join(self.best_checkpoint_path, f'{self.model_name}_best.pth'), val_loader, dataset='val')

    def validate(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            tbar = tqdm(val_loader, ncols=80)
            val_epoch_loss = []
            for idx, (val_stmap, val_rsp_label) in enumerate(tbar):
                val_stmap = val_stmap.to(self.device)
                val_rsp_label = val_rsp_label.to(self.device)
                rsp_pred = self.model(val_stmap)
                val_loss = self.criterion(rsp_pred, val_rsp_label)
                val_epoch_loss.append(val_loss.item())
                tbar.set_postfix(val_loss=val_loss.item())
            self.val_loss_ls.append(sum(val_epoch_loss)/len(val_epoch_loss))
            print(f"Val Loss: {sum(val_epoch_loss) / len(val_epoch_loss)}")
    

    def inference(self, model_path, data_loader, dataset='train'):
        # 输出pickle文件
        _model = self._load_state_dict(model_path)
        data = {
            'pred': [],
            'label': []
        }
        with torch.no_grad():
            tbar = tqdm(data_loader, ncols=80)
            for idx, (stmap, rsp_label) in enumerate(tbar):
                stmap = stmap.to(self.device)
                rsp_pred = _model(stmap) 
                for i in range(rsp_pred.shape[0]):
                    data['pred'].append(rsp_pred[i])
                    data['label'].append(rsp_label[i])
        with open(os.path.join(self.best_checkpoint_path, f'{self.model_name}_pred_{dataset}.pkl'), 'wb') as f:
            pickle.dump(data, f)
            print(f"Prediction saved at {self.best_checkpoint_path}/{self.model_name}_pred_{dataset}.pkl")
