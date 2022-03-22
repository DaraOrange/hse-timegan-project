import numpy as np
import torch
from model import Embedder


def MinMaxScaler(dataX):

    min_val = np.min(np.min(dataX, axis = 0), axis = 0)
    dataX = dataX - min_val

    max_val = np.max(np.max(dataX, axis = 0), axis = 0)
    dataX = dataX / (max_val + 1e-7)

    return dataX, min_val, max_val


#%% Start TGAN function (Input: Original data, Output: Synthetic Data)
def train(dataX, parameters):
	# tf.reset_default_graph()  # https://discuss.pytorch.org/t/how-to-free-graph-manually/9255

	# Basic Parameters
    No = len(dataX)
    data_dim = len(dataX[0][0,:])

    # Maximum seq length and each seq length
    dataT = list()
    Max_Seq_Len = 0
    for i in range(No):
        Max_Seq_Len = max(Max_Seq_Len, len(dataX[i][:,0]))
        dataT.append(len(dataX[i][:,0]))

    # Normalization
    if ((np.max(dataX) > 1) | (np.min(dataX) < 0)):
        dataX, min_val, max_val = MinMaxScaler(dataX)
        Normalization_Flag = 1
    else:
        Normalization_Flag = 0

    # Network Parameters
    hidden_dim   = parameters['hidden_dim']
    num_layers   = parameters['num_layers']
    iterations   = parameters['iterations']
    batch_size   = parameters['batch_size']
    module_name  = parameters['module_name']    # 'lstm' or 'lstmLN'
    z_dim        = parameters['z_dim']
    gamma        = 1
    lr 			 = parameters['lr']

    # input place holders  # https://discuss.pytorch.org/t/placeholder-in-pytorch/96614

    # X = tf.placeholder(tf.float32, [None, Max_Seq_Len, data_dim], name = "myinput_x")
    # Z = tf.placeholder(tf.float32, [None, Max_Seq_Len, z_dim], name = "myinput_z")
    # T = tf.placeholder(tf.int32, [None], name = "myinput_t")

    # functions declaration -> classes in model.py

    #%% Random vector generation
    def random_generator (batch_size, z_dim, T_mb, Max_Seq_Len):

        Z_mb = list()

        for i in range(batch_size):

            Temp = np.zeros([Max_Seq_Len, z_dim])

            Temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])

            Temp[:T_mb[i],:] = Temp_Z

            Z_mb.append(Temp_Z)

        return Z_mb

    # G_loss_U, G_loss_S, G_loss_V

    # Loss for the generator
    # 1. Adversarial loss
    G_loss_U = (
        lambda Y_fake:
        torch.nn.functional.binary_cross_entropy_with_logits(
            Y_fake,
            torch.ones_like(Y_fake)
        )
    )
    # G_loss_U_e = lambda Y_fake_e: torch.nn.functional.binary_cross_entropy_with_logits(Y_fake_e, torch.ones_like(Y_fake_e))

    # 2. Supervised loss
    G_loss_S = (
        lambda H_hat_supervise, H:
        torch.nn.functional.mse_loss(H_hat_supervise[:,:-1,:], H[:,1:,:])
    )

    # 3. Two Momments
    G_loss_V1 = (
        lambda X_hat, X:
        torch.mean(torch.abs(
            torch.sqrt(X_hat.var(dim=0, unbiased=False) + 1e-6) -
            torch.sqrt(X.var(dim=0, unbiased=False) + 1e-6)
        ))
    )
    #tf.reduce_mean(np.abs(tf.sqrt(tf.nn.moments(X_hat,[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X,[0])[1] + 1e-6)))
    G_loss_V2 = torch.mean(torch.abs((X_hat.mean(dim=0)) - (X.mean(dim=0))))
    #tf.reduce_mean(np.abs((tf.nn.moments(X_hat,[0])[0]) - (tf.nn.moments(X,[0])[0])))

    # Loss for the embedder network
    E_loss_T0 = torch.nn.functional.mse_loss
    E_loss0 = lambda X, X_tilde: 10 * torch.sqrt(E_loss_T0(X, X_tilde))
    E_loss = lambda X, X_tilde: E_loss0(X, X_tilde) + 0.1*G_loss_S()

    embedder = Embedder(..., hidden_dim, num_layers)
    embedder.to(parameters['device'])

    E0_opt = torch.optim.Adam(embedder.parameters(), lr=lr)
    E_opt = torch.optim.Adam(recovery.parameters(), lr=lr)
    D_opt = torch.optim.Adam(discriminator.parameters(), lr=lr)
    G_opt = torch.optim.Adam(generator.parameters(), lr=lr)
    GS_opt = torch.optim.Adam(supervisor.parameters(), lr=lr)

    for it in range(iterations):
    	# Batch setting
        idx = np.random.permutation(No)
        train_idx = idx[:batch_size]

        X_mb = list(dataX[i] for i in train_idx)
        T_mb = list(dataT[i] for i in train_idx)

        step_g_loss_s = E_loss(X_mb, T_mb)
        step_g_loss_s.backward()
        E0_opt.step()

   	for it in range(iterations):

        # Generator Training
        for kk in range(2):

            # Batch setting
            idx = np.random.permutation(No)
            train_idx = idx[:batch_size]

            X_mb = list(dataX[i] for i in train_idx)
            T_mb = list(dataT[i] for i in train_idx)

            # Random vector generation
            Z_mb = random_generator(batch_size, z_dim, T_mb, Max_Seq_Len


            G_loss_U, G_loss_S, G_loss_V
           	G_opt.step()
