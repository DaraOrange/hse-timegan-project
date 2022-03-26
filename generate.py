import numpy as np
import torch
from model import Embedder, Generator, Discriminator, Recovery, Supervisor
from data_loading import scale
import argparse
import wandb

def generate(dataX, parameters):
    os.environ["WANDB_API_KEY"] = "fb8e2b4baa2d663fead9edbdd132da5080eeb6ee"
    os.environ["WANDB_MODE"] = "online"

    wandb.init(project="TimeGAN")

    No = len(dataX)
    data_dim = len(dataX[0][0,:])

    dataT = list()
    Max_Seq_Len = 0
    for i in range(No):
        Max_Seq_Len = max(Max_Seq_Len, len(dataX[i][:,0]))
        dataT.append(len(dataX[i][:,0]))

    min_val = np.min(np.min(dataX, axis = 0), axis = 0)
    max_val = np.max(np.max(dataX, axis = 0), axis = 0)
    if ((np.max(max_val) > 1) | (np.min(min_val) < 0)):
        dataX = scale(dataX)
        Normalization_Flag = 1
    else:
        Normalization_Flag = 0

    hidden_size  = parameters['hidden_size']
    num_layers   = parameters['num_layers']
    iterations   = parameters['iterations']
    batch_size   = parameters['batch_size']
    z_dim        = data_dim
    gamma        = 1
    lr 			 = parameters['lr']
    device       = parameters['device']

    def random_generator(batch_size, z_dim, T_mb, Max_Seq_Len):
        Z_mb = list()
        for i in range(batch_size):
            Temp = np.zeros([Max_Seq_Len, z_dim])
            Temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
            Temp[:T_mb[i],:] = Temp_Z
            Z_mb.append(Temp_Z)
        return Z_mb

    embedder = Embedder(data_dim, hidden_size, num_layers).to(device)
    discriminator = Discriminator(hidden_size, num_layers).to(device)
    generator = Generator(z_dim, hidden_size, num_layers).to(device)
    recovery = Recovery(hidden_size, data_dim, num_layers).to(device)
    supervisor = Supervisor(hidden_size, num_layers).to(device)

    e_opt = torch.optim.Adam(embedder.parameters(), lr)
    r_opt = torch.optim.Adam(recovery.parameters(), lr)
    s_opt = torch.optim.Adam(supervisor.parameters(), lr)
    g_opt = torch.optim.Adam(generator.parameters(), lr)
    d_opt = torch.optim.Adam(discriminator.parameters(), lr)

    print('Start Embedding Network Training')

    for itt in range(iterations):
        e_opt.zero_grad()
        r_opt.zero_grad()
        s_opt.zero_grad()
        g_opt.zero_grad()
        d_opt.zero_grad()

        idx = np.random.permutation(No)
        train_idx = idx[:batch_size]

        X = torch.tensor(np.array(list(dataX[i] for i in train_idx)), dtype=torch.float).to(device)
        T = torch.tensor(np.array(list(dataT[i] for i in train_idx)), dtype=torch.int64)

        H = embedder(X, T)
        X_tilde = recovery(H, T)

        H_hat_supervise = supervisor(H, T)

        G_loss_S = torch.nn.MSELoss()(H[:,1:,:], H_hat_supervise[:,1:,:])
        E_loss_T0 = torch.nn.MSELoss()(X, X_tilde)
        E_loss0 = 10*torch.sqrt(E_loss_T0)
        E_loss = E_loss0  + 0.1*G_loss_S

        E_loss0.backward()

        e_opt.step()
        r_opt.step()

        wandb.log({'Embedder loss': np.sqrt(step_e_loss)})
        if itt % 1000 == 0:
            print('step: '+ str(itt) + ', e_loss: ' + str(round(E_loss0.item(), 8)))

    print('Finish Embedding Network Training')

    print('Start Training with Supervised Loss Only')

    for itt in range(iterations):
        e_opt.zero_grad()
        r_opt.zero_grad()
        s_opt.zero_grad()
        g_opt.zero_grad()
        d_opt.zero_grad()

        idx = np.random.permutation(No)
        train_idx = idx[:batch_size]

        X = torch.tensor(np.array(list(dataX[i] for i in train_idx)), dtype=torch.float).to(device)
        T = torch.tensor(np.array(list(dataT[i] for i in train_idx)), dtype=torch.int64)

        H = embedder(X, T)
        H_hat_supervise = supervisor(H, T)
        G_loss_S = torch.nn.MSELoss()(H[:,1:,:], H_hat_supervise[:,1:,:])
        G_loss_S.backward()
        s_opt.step()

        wandb.log({'Supervisor loss': np.sqrt(np.sqrt(step_g_loss_s))})
        if itt % 1000 == 0:
            print('step: '+ str(itt) + ', s_loss: ' + str(np.round(np.sqrt(G_loss_S.item()),4)) )

    print('Finish Training with Supervised Loss Only')

    print('Start Joint Training')

    for itt in range(iterations):

        idx = np.random.permutation(No)
        train_idx = idx[:batch_size]

        X = torch.tensor(np.array(list(dataX[i] for i in train_idx)), dtype=torch.float).to(device)
        T = torch.tensor(np.array(list(dataT[i] for i in train_idx)), dtype=torch.int64)

        for _ in range(2):
            e_opt.zero_grad()
            r_opt.zero_grad()
            s_opt.zero_grad()
            g_opt.zero_grad()
            d_opt.zero_grad()

            H = embedder(X, T)
            H_hat_supervise = supervisor(H, T)
            X_tilde = recovery(H, T)

            Z = torch.rand((batch_size, Max_Seq_Len, z_dim)).to(device)
            E_hat = generator(Z, T)
            H_hat = supervisor(E_hat, T)

            X_hat = recovery(H_hat, T)

            Y_fake = discriminator(H_hat, T)
            Y_fake_e = discriminator(E_hat, T)

            G_loss_U = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake, torch.ones_like(Y_fake))
            G_loss_U_e = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake_e, torch.ones_like(Y_fake_e))

            G_loss_S = torch.nn.functional.mse_loss(H_hat_supervise[:,:-1,:], H[:,1:,:])

            G_loss_V1 = torch.mean(torch.abs(torch.sqrt(X_hat.var(dim=0, unbiased=False) + 1e-6) - torch.sqrt(X.var(dim=0, unbiased=False) + 1e-6)))
            G_loss_V2 = torch.mean(torch.abs((X_hat.mean(dim=0)) - (X.mean(dim=0))))

            G_loss_V = G_loss_V1 + G_loss_V2

            G_loss = G_loss_U + gamma * G_loss_U_e + 100 * torch.sqrt(G_loss_S) + 100 * G_loss_V

            G_loss.backward()
            g_opt.step()
            s_opt.step()


            H = embedder(X, T)
            X_tilde = recovery(H, T)

            H_hat_supervise = supervisor(H, T)
            G_loss_S = torch.nn.functional.mse_loss(
                H_hat_supervise[:,:-1,:],
                H[:,1:,:]
            )

            E_loss_T0 = torch.nn.functional.mse_loss(X_tilde, X)
            E_loss0 = 10 * torch.sqrt(E_loss_T0)
            E_loss = E_loss0 + 0.1 * G_loss_S

            E_loss.backward()
            e_opt.step()
            r_opt.step()


        Z = torch.rand((batch_size, Max_Seq_Len, z_dim)).to(device)

        H = embedder(X, T).detach()
        H_hat = supervisor(H, T).detach()
        E_hat = generator(Z, T).detach()

        Y_real = discriminator(H, T)
        Y_fake = discriminator(H_hat, T)
        Y_fake_e = discriminator(E_hat, T)

        D_loss_real = torch.nn.functional.binary_cross_entropy_with_logits(Y_real, torch.ones_like(Y_real))
        D_loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake, torch.zeros_like(Y_fake))
        D_loss_fake_e = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake_e, torch.zeros_like(Y_fake_e))

        D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

        D_loss.backward()
        d_opt.step()

        wandb.log({'Disc loss': step_d_loss,
               'G_loss_u': step_g_loss_u,
               'g_loss_s': np.sqrt(step_g_loss_s),
               'g_loss_v': step_g_loss_v,
               'e_loss_t0': np.sqrt(step_e_loss_t0)})

        if itt % 1000 == 0:
            print('step: '+ str(itt) + '/' + str(iterations) +
                    ', d_loss: ' + str(np.round(D_loss.item(), 8)) +
                    ', g_loss_u: ' + str(np.round(G_loss_U.item(), 8)) +
                    ', g_loss_s: ' + str(np.round(np.sqrt(G_loss_S.item()), 8)) +
                    ', g_loss_v: ' + str(np.round(G_loss_V.item(), 8)) +
                    ', e_loss_t0: ' + str(np.round(np.sqrt(E_loss_T0.item()), 8)))


    print('Finish Joint Training')

    Z = torch.rand((batch_size, Max_Seq_Len, z_dim)).to(device)
    E_hat = generator(Z, T)
    H_hat = supervisor(E_hat, T)
    generated_data_curr = recovery(H_hat, T)

    generated_data = list()

    for i in range(batch_size):
        temp = generated_data_curr[i,:dataT[i],:].cpu().detach().numpy()
        generated_data.append(temp)

    if Normalization_Flag:
      generated_data = generated_data * max_val
      generated_data = generated_data + min_val

    return generated_data
