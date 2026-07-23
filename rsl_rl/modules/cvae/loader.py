import torch
import numpy as np

# np.set_printoptions(threshold=np.inf)

class Loader:
    def __init__(self, paths, device):
        if device == 'cuda:0':
            self.device = torch.device(device)
        else:
            self.device = device
        
        self.states, self.height_maps = self.load_data(paths)
        print(f'Total data length {len(self.states)}, states shape {self.states[0].shape}, height_maps shape {self.height_maps[0].shape}')
    
    
    def load_data(self, paths):
        states = []
        height_maps = []
        num_data = len(paths)
        # 3+3+26+26+12=70
        required_keys = ['base_lin_vel', 'base_ang_vel', 'joint_pos', 'joint_vel', 'eef_pos_body', 'height_scan']
        for data_id in range(num_data):
            data_dict = np.load(paths[data_id])
            print(f'Load data from file {paths[data_id]}')
            keys = list(data_dict.keys())
            # print(keys)
            # exit(0)
            num_env = len(keys) // 10
            for env_id in range(num_env):
                for required_key in required_keys:
                    required_key = f'{env_id}_{required_key}'
                    assert required_key in keys, f'{required_key} is not in dataset'
                
                base_vel = torch.tensor(data_dict[f'{env_id}_base_lin_vel'], dtype=torch.float32, device=self.device)
                base_ang = torch.tensor(data_dict[f'{env_id}_base_ang_vel'], dtype=torch.float32, device=self.device)
                joint_pos = torch.tensor(data_dict[f'{env_id}_joint_pos'], dtype=torch.float32, device=self.device)
                joint_vel = torch.tensor(data_dict[f'{env_id}_joint_vel'], dtype=torch.float32, device=self.device)
                eef_pos = torch.tensor(data_dict[f'{env_id}_eef_pos_body'], dtype=torch.float32, device=self.device)
                states.append(torch.cat([base_vel, base_ang, joint_pos, joint_vel, eef_pos], dim=1))
                
                height_map = torch.tensor(data_dict[f'{env_id}_height_scan'], dtype=torch.float32, device=self.device)
                height_maps.append(height_map)
        
        return states, height_maps
        

    def batch_sample(self, batch_size, seq_len, symmetry):
        num_per_data = batch_size // len(self.states)
        seq = torch.arange(0, seq_len, dtype=torch.int, device=self.device)
        batch_states = []
        batch_height_maps = []
        
        # average sample
        for i in range(len(self.states)):
            data_length = self.states[i].shape[0]
            max_start = data_length - seq_len
            assert max_start > 0, f'The length of dataset {i} less than seq_len'
            
            start = torch.randint(0, max_start, (num_per_data,), device=self.device)
            frame = start.reshape(-1, 1) + seq.reshape(1, -1)
            batch_states.append(self.states[i][frame])
            batch_height_maps.append(self.height_maps[i][frame])
        
        # sample the rest
        rest_length = batch_size - num_per_data*len(self.states)
        for i in range(rest_length):
            data_length = self.states[i].shape[0]
            max_start = data_length - seq_len
            
            start = torch.randint(0, max_start, (1,), device=self.device)
            frame = start.reshape(-1, 1) + seq.reshape(1, -1)
            batch_states.append(self.states[i][frame])
            batch_height_maps.append(self.height_maps[i][frame])
        
        return torch.cat(batch_states, dim=0), torch.cat(batch_height_maps, dim=0)
    
    
if __name__ == '__main__':
    paths = [
        '../resources/cvae/data_1211_flat_slop_stair/upstairs_H10W30_rolloutdata_2025-12-11_20-34-55/data.npz',
    ]
    loader = Loader(paths, 'cpu')
    # states, height_maps = loader.batch_sample(128, 64, False)
    # print(f'states shape {states.shape}')
    # print(f'height_maps shape {height_maps.shape}')
    states, height_maps = loader.batch_sample(64, 32, False)
    print(height_maps[0][1])

    # print(f'states shape {states.shape}')
    # print(f'height_maps shape {height_maps.shape}')

    # print(states.permute(0,2,1).shape)
    # print(height_maps.permute(0,2,1).shape)

    # print(f'states shape {states.shape}')
    # print(f'height_maps shape {height_maps.shape}')

    # print(torch.cat([states.permute(0,2,1), height_maps.permute(0,2,1)], dim=1).shape)
    
    
    
