import torch
import torch.nn as nn
import torch.nn.functional as F


class VaeModel(nn.Module):
    def __init__(
        self,
        seq_len,
        input_dim,
        condition_dim,
        base_dim,
        latent_dim,
        hidden_dims, # [128,64,32]
        alpha,
        beta
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.beta = beta
        self.cur_beta = 0

        self.height_map_cnn = HeightMapCNN()
        self.input_conv1 = nn.Conv1d(input_dim+condition_dim, base_dim, kernel_size=3, stride=1, padding=1)
        encoder_layer = []
        
        for i, hidden_dim in enumerate(hidden_dims):
            encoder_layer.append(
                ResBlock(base_dim, hidden_dim, alpha)
            )
            
            if i != len(hidden_dims) - 1:
                encoder_layer.append(
                    Down(hidden_dim)
                )

                seq_len = (seq_len+1) // 2
            
            base_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layer)
        
        self.bottom_seq_len = seq_len
        self.bottom_base_dim = base_dim
        
        self.flatten = nn.Sequential(
            nn.Linear(self.bottom_seq_len*self.bottom_base_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.LeakyReLU(alpha)
        )
        
        self.mu_fc = nn.Linear(latent_dim, latent_dim)
        self.var_fc = nn.Linear(latent_dim, latent_dim)
        
        self.squeeze = nn.Sequential(
            nn.Linear(latent_dim+condition_dim, self.bottom_seq_len*self.bottom_base_dim),
            nn.LayerNorm(self.bottom_seq_len*self.bottom_base_dim),
            nn.LeakyReLU(alpha)
        )
        
        decoder_layer = []
        reverse_hidden_dims = list(reversed(hidden_dims))
        
        for i, reverse_hidden_dim in enumerate(reverse_hidden_dims):
            decoder_layer.append(
                ResBlock(base_dim, reverse_hidden_dim, alpha)
            )
            
            if i != len(reverse_hidden_dims) - 1:
                decoder_layer.append(
                    Up(reverse_hidden_dim)
                )

                seq_len *= 2
            
            base_dim = reverse_hidden_dim
        
        self.decoder = nn.Sequential(*decoder_layer)
        
        self.output_conv1 = nn.Sequential(
            nn.Conv1d(base_dim, input_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(alpha)
        )
        self.adjuster = nn.Linear(input_dim*seq_len, input_dim*self.seq_len)
    
    def encode(self, x, c):
        c = self.height_map_cnn(c)
        x = torch.cat([x, c], dim=1)
        B = x.shape[0]
        x = self.input_conv1(x)
        x = self.encoder(x)
        x = x.reshape(B, -1)
        x = self.flatten(x)
        
        mu = self.mu_fc(x)
        log_var = self.var_fc(x)
        
        return mu, log_var
    
    def reparam(self, mu, log_var):
        e = torch.randn_like(log_var)
        
        return mu + torch.exp(0.5*log_var)*e
    
    def decode(self, x, c):
        c = self.height_map_cnn(c.unsqueeze(2)).squeeze(2)
        x = torch.cat([x, c], dim=1)
        B = x.shape[0]
        x = self.squeeze(x)
        x = x.reshape(B, self.bottom_base_dim, self.bottom_seq_len)
        x = self.decoder(x)
        
        x = self.output_conv1(x)
        x = x.reshape(B, -1)
        x = self.adjuster(x)
        x = x.reshape(B, self.input_dim, self.seq_len)
        
        return x
    
    def forward(self, x, c):
        mu, log_var = self.encode(x, c)
        z = self.reparam(mu, log_var)
        recon_x = self.decode(z, c[:,:,0])
        
        return recon_x, x, z, mu, log_var
    
    def compute_loss(self, recon_x, x, z, mu, log_var):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + log_var - mu**2 - torch.exp(log_var))
        
        loss = {
            'loss': recon_loss + self.cur_beta * kl_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'beta': self.cur_beta
        }
        
        return loss
    
    def beta_schedule(self, step, total_step):
        self.cur_beta = min(10.0*(step/total_step), 1.0)*self.beta


class ResBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        alpha
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.ac1 = nn.LeakyReLU(alpha)

        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.ac2 = nn.LeakyReLU(alpha)

        if input_dim != output_dim:
            self.sc = nn.Conv1d(input_dim, output_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.sc = nn.Identity()

    def forward(self, x):
        r = self.ac1(self.bn1(self.conv1(x)))
        r = self.bn2(self.conv2(r))
        x = self.sc(x)
        x = self.ac2(r+x)

        return x


class Down(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, input_dim, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        return self.conv1(x)
    

class Up(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, input_dim, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv1(x)


class HeightMapCNN(nn.Module):
    def __init__(self, H=17, W=11, output_dim=64):
        super().__init__()

        self.H = H
        self.W = W

        self.conv2 = nn.Sequential(
            # [B*T, 1, 17, 11] -> [B*T, 32, 17, 11]
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            # [B*T, 32, 17, 11] -> [B*T, 64, 9, 6]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

        # 消除空间维度
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, output_dim)
        self.output_dim = output_dim

    def forward(self, height_map):
        """
        height_map: [B, 187, T]
        return:     [B, 64,  T]
        """
        B, C, T = height_map.shape
        min_height = height_map.min(dim=1, keepdim=True).values
        height_map = height_map - min_height
        h = height_map.permute(0, 2, 1)
        h = h.reshape(B * T, 1, self.W, self.H)
        h = h.permute(0, 1, 3, 2).contiguous()

        h = self.conv2(h)                    # [B*T, 64, h', w']
        h = self.pool(h).view(B * T, -1)     # [B*T, 64]
        h = self.fc(h)                       # [B*T, 64]

        h = h.view(B, T, -1).permute(0, 2, 1)

        return h



