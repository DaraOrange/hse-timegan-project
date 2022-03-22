import numpy as np
from sklearn.metrics import mean_absolute_error
from utils import extract_time
import torch
from torch import nn


class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        self.p_cell = nn.Sequential(*[
            nn.GRU(input_dim=input_dim, hidden_dim=hidden_dim),
            nn.Tanh()
        ])
        self.linear = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def __call__(self, x, t):
        x_packed = nn.utils.rnn.pack_padded_sequence(
            input=x, 
            lengths=t, 
            batch_first=True, 
            enforce_sorted=False
        )
        x_packed = x_packed.cuda()
        _, p_last_state = self.p_cell(x_packed)
        y_hat_logit = self.linear(p_last_state)
        y_hat = self.sigmoid(y_hat_logit)
        return y_hat


def predictive_score_metrics (ori_data, generated_data):
    no, seq_len, dim = np.asarray(ori_data).shape
      
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(ori_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])  
      
    hidden_dim = int(dim/2)
    iterations = 5000
    batch_size = 128

    predictor = Predictor(dim, hidden_dim).cuda()
      
    opt = torch.optim.Adam(predictor.parameters())
      
    for itt in range(iterations):
        opt.zero_grad()
        idx = np.random.permutation(len(generated_data))
        train_idx = idx[:batch_size]     
                
        X_mb = list(generated_data[i][:-1,:(dim-1)] for i in train_idx)
        T_mb = list(generated_time[i]-1 for i in train_idx)
        Y_mb = list(np.reshape(generated_data[i][1:,(dim-1)],[len(generated_data[i][1:,(dim-1)]),1]) for i in train_idx)  

        y_pred = predictor(X_mb, T_mb)

        p_loss = nn.L1Loss()(Y_mb, y_pred)      
        p_loss.backward()
        opt.step()     
    
    predictor.eval()

    idx = np.random.permutation(len(ori_data))
    train_idx = idx[:no]
      
    X_mb = list(ori_data[i][:-1,:(dim-1)] for i in train_idx)
    T_mb = list(ori_time[i]-1 for i in train_idx)
    Y_mb = list(np.reshape(ori_data[i][1:,(dim-1)], [len(ori_data[i][1:,(dim-1)]),1]) for i in train_idx)
      
    pred_Y_curr = predictor(X_mb, T_mb)
      
    MAE_temp = 0
    for i in range(no):
      MAE_temp = MAE_temp + mean_absolute_error(Y_mb[i], pred_Y_curr[i,:,:])
      
    predictive_score = MAE_temp / no
      
    return predictive_score
