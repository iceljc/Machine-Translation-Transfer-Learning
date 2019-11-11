import torch

params = {}

### general parameters
params['USE_CUDA'] = torch.cuda.is_available()
params['DEVICE'] = torch.device('cuda:0')

params['embed_size'] = 300
params['hidden_size'] = 256
params['num_layers'] = 1
params['dropout'] = 0.2


params['num_epochs'] = 30
params['learning_rate'] = 0.0003
params['batch_size'] = 64