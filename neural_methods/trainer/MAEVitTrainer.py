import torch
import torch.nn as nn
from tqdm import tqdm
from neural_methods.utils.select_model import get_model
import os
import matplotlib.pyplot as plt
import time

class MAEVitTrainer:
    def __init__(self, configs):
        self.configs = configs

        self.model_name = configs.MODEL.NAME
        self.model = get_model(configs)
        if configs.TRAIN.CONTINUE_TRAINING and configs.MODEL.MODEL_PATH:
            self.model.load_state_dict(torch.load(configs.MODEL.MODEL_PATH))
            print(f"Model {self.model_name} loaded from {configs.MODEL.MODEL_PATH}")

        self.mask_ratio = configs.MODEL.MAE.MASK_RATIO
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion.to(self.device)
        self.min_val_loss = float('inf')
        
        self.train_loss = []
        self.val_loss = []
        # 获取当前时间字符串yyyymmddhhmm
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

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, f'{self.model_name}_{epoch}.pth'))
        print(f"Model {self.model_name} saved at epoch {epoch}")

    def save_best_model(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.best_checkpoint_path, f'{self.model_name}_best.pth'))
        print(f"Best model {self.model_name} saved at epoch {epoch}")

    def save_loss_plot(self):
        # 清除之前的图
        plt.clf()
        plt.plot(self.train_loss, label='train')
        plt.plot(self.val_loss, label='val')
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
            for idx, (stmap, _ ) in enumerate(tbar):
                tbar.set_description(f"Epoch {epoch+1}/{self.epochs}")
                stmap = stmap.to(self.device)
                
                self.optimizer.zero_grad()
                loss, BVP_map, mask = self.model(stmap,self.mask_ratio)
                loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                tbar.set_postfix(loss=loss.item())
                train_epoch_loss.append(loss.item())
            self.validate(val_loader)
            print(f"Train Loss: {sum(train_epoch_loss) / len(train_epoch_loss)}")
            print(f"Time Elapsed: {time.time() - start_time}")
            self.train_loss.append(sum(train_epoch_loss) / len(train_epoch_loss))
            if epoch % self.configs.MODEL.SAVE_INTERVAL == 0:
                self.min_val_loss = self.val_loss[-1]
                self.save_model(epoch)
            if self.val_loss[-1] < self.min_val_loss:
                self.min_val_loss = self.val_loss[-1]
                self.save_best_model(epoch)

            self.save_loss_plot()
    
    def validate(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            tbar = tqdm(val_loader, ncols=80)
            val_epoch_loss = []
            for idx, (stmap, _) in enumerate(tbar):

                stmap = stmap.to(self.device)
                loss, BVP_map, mask = self.model(stmap, self.mask_ratio)
                val_epoch_loss.append(loss.item())
                tbar.set_postfix(val_loss=loss.item())
            self.val_loss.append(sum(val_epoch_loss)/len(val_epoch_loss))
            print(f"Val Loss: {sum(val_epoch_loss) / len(val_epoch_loss)}")
            
