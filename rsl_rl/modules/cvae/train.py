
import os
import wandb
import torch
import torch.optim as optim
import datetime


from config import VaeConfig
from model import VaeModel
from loader import Loader


class VaeTrain:
    step_counter = 0
    
    def __init__(self, cfg=VaeConfig):
        self.cfg = cfg
        self.device = torch.device(self.cfg.train.device)
        self.symmetry = self.cfg.train.use_symmetry
        
        self.model = VaeModel(
            self.cfg.model.seq_len,
            self.cfg.model.input_dim,
            self.cfg.model.condition_dim,
            self.cfg.model.base_dim,
            self.cfg.model.latent_dim,
            self.cfg.model.hidden_dims,
            self.cfg.model.alpha,
            self.cfg.model.beta
        ).to(self.device)
        
        self.data_loader = Loader(self.cfg.data.train_data_paths, self.device)
        self.optim = optim.Adam(self.model.parameters(), lr=self.cfg.train.lr)
        
        if self.cfg.train.resume:
            param_dict = torch.load(self.cfg.train.load_run, weights_only=True)
            self.model.load_state_dict(param_dict['model_state_dict'])
            # self.optim.load_state_dict(param_dict['optim_state_dict'])
            print(f'Load model from {self.cfg.train.load_run}')
        
        self.now = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        self.sub_dir = self.cfg.train.run_name+ '-' + self.now
        self.log_dir = os.path.join(self.cfg.train.log_dir, self.sub_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        
    def train(self):
        wandb.init(
            project=self.cfg.train.run_name,
            name=self.now
        )
        
        self.model.train()
        for step in range(self.cfg.train.train_time):
            if not self.symmetry:
                batch_states, batch_height_maps = self.data_loader.batch_sample(
                    self.cfg.train.batch_size,
                    self.cfg.model.seq_len,
                    self.symmetry
                )
                batch_states = batch_states.permute(0,2,1)
                batch_height_maps = batch_height_maps.permute(0,2,1)
            else:
                pass
            
            recon_x, x, z, mu, log_var = self.model(batch_states, batch_height_maps)
            self.mu = mu
            self.log_var = log_var
            
            if self.symmetry:
                pass
            
            loss_dict = self.model.compute_loss(recon_x, x, z, mu, log_var)
            
            loss = loss_dict['loss']
            if self.symmetry:
                pass
            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
            self.step_counter = step + 1
            wandb.log(loss_dict, step=self.step_counter)
            if self.step_counter % self.cfg.train.save_interval == 0:
                self.save()
            
            self.model.beta_schedule(self.step_counter, self.cfg.train.train_time)
    
    
    def save(self):
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            # 'optim_state_dict': self.optim.state_dict(),
            # 'mu': self.mu,
            # 'log_var': self.log_var
        }
        
        save_name = f'model_{self.step_counter}.pt'
        save_path = os.path.join(self.log_dir, save_name)
        
        torch.save(save_dict, save_path)
    
    
    def generator(self, batch_size):
        self.model.eval()
        z = torch.randn(batch_size, self.model.latent_d).to(self.device)
        x = self.model.decode(z)
        
        return x


if __name__ == '__main__':
    cfg = VaeConfig()
    trainer = VaeTrain(cfg)
    trainer.train()
