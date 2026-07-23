import os

class VaeConfig:
    class data:
        # train_data_dir = '/home/hitcsc/Downloads/test'
        # train_data_paths = []
        # for dir in os.listdir(train_data_dir):
        #     if os.path.isdir(os.path.join(train_data_dir, dir)):
        #         train_data_paths.append(os.path.join(train_data_dir, dir, 'data.npz'))
        test_data_path = './test/data.npz'
    
    class model:
        seq_len = 16
        input_dim = 70
        condition_dim = 64
        base_dim = 128
        latent_dim = 64
        hidden_dims = [128, 64, 32]
        alpha = 0.01
        beta = 0.7
    
    class train:
        run_name = 'cvae'
        device = 'cuda:0'
        lr = 1e-4
        train_time = 1000000
        batch_size = 1024
        save_interval = 10000
        use_symmetry = False
        log_dir = './log'
        
        resume = False
        load_run = ''
    
    class test:
        batch_size = 1
        load_path = '/home/sakura/kuavo/kuavo-emp/cvae/log/baseline/model_200000_03_06.pt'
        condition_path = ''
        
        
        