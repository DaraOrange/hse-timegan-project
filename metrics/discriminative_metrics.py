import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score
from utils import train_test_divide, extract_time, batch_generator


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=1):
        self.d_cell = nn.GRU(input_dim=input_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.tanh = nn.Tanh()
        self.logit = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid() 

    def __call__(self, x, T):
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=x, 
            lengths=T, 
            batch_first=True, 
            enforce_sorted=False
        )
        x_packed = x_packed.cuda()
        _, d_last_state = self.d_cell(x_packed)
        d_last_state = self.tanh(d_last_state)
        logit = self.logit(d_last_state).squeeze()  
        return logit, self.sigmoid(logit)


def disc_loss(discriminator, X, T, X_hat, T_hat):
    y_logit_real, y_pred_real = discriminator(X, T)
    y_logit_fake, y_pred_fake = discriminator(X_hat, T_hat)
            
    d_loss_real = nn.BCEWithLogitsLoss()(y_logit_real, torch.ones_like(y_logit_real))
    d_loss_fake = nn.BCEWithLogitsLoss()(y_logit_fake, labels = torch.zeros_like(y_logit_fake))
    d_loss = d_loss_real + d_loss_fake  

    return d_loss, y_pred_real, y_pred_fake


def discriminative_score_metrics (ori_data, generated_data):
    no, seq_len, dim = np.asarray(ori_data).shape    
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(ori_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])  
        
    hidden_dim = int(dim/2)
    iterations = 2000
    batch_size = 128

    discriminator = Discriminator(input_dim=dim, hidden_dim=hidden_dim).cuda()
    opt = torch.optim.Adam(discriminator.parameters())
        
    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
        train_test_divide(ori_data, generated_data, ori_time, generated_time)
        
    for _ in range(iterations):
        opt.zero_grad()

        X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
        X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)
            
        d_loss, _, _ = disc_loss(discriminator, X=X_mb, T=T_mb, X_hat=X_hat_mb, T_hat=T_hat_mb)
        d_loss.backward()
        opt.step()       
        
    discriminator.eval()
    _, y_pred_real_curr, y_pred_fake_curr = disc_loss(discriminator, X=test_x, T=test_t, X_hat=test_x_hat, T_hat=test_t_hat)
    y_pred_real_curr = y_pred_real_curr.cpu().numpy()
    y_pred_fake_curr = y_pred_fake_curr.cpu().numpy()
        
    y_pred_final = np.squeeze(np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis = 0))
    y_label_final = np.concatenate((np.ones([len(y_pred_real_curr),]), np.zeros([len(y_pred_fake_curr),])), axis = 0)
        
    acc = accuracy_score(y_label_final, (y_pred_final>0.5))
    discriminative_score = np.abs(0.5-acc)
        
    return discriminative_score
