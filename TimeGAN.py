import torch
from torch.nn import Module, GRU, Linear, Sigmoid

class Generator(Module):
    def __init__(self, input_size, output_size, num_layers):
        """Noise sequence -> Original space"""
        self.rnn = GRU(input_size=input_size, 
                       hidden_size=output_size, 
                       num_layers=num_layers, 
                       batch_first=True)
        self.linear = Linear(in_features=output_size, out_features=output_size)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        # [B x T x Z] -> [B x T x H]
        output, _ = self.rnn(x)
        output = self.linear(x)
        output = self.sigmoid(output)
        return output

class Discriminator(Module):
    """Original space -> logits"""
    def __init__(self, hidden_size, num_layers):
        self.rnn = GRU(input_size=hidden_size, 
                       hidden_size=hidden_size, 
                       num_layers=num_layers, 
                       batch_first=True)
        self.linear = Linear(in_features=hidden_size, out_features=1)

    def forward(self, x):
        # [B x T x F] -> [B x T x 1]
        output, _ = self.rnn(x)
        output = self.linear(x).squeeze(-1)
        return output

class Embedder(Module):
    """Original space -> Embedding space"""
    def __init__(self, input_size, hidden_size, num_layers):
        self.rnn = GRU(input_size=input_size, 
                       hidden_size=hidden_size, 
                       num_layers=num_layers, 
                       batch_first=True)
        self.linear = Linear(in_features=hidden_size, out_features=hidden_size)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        # [B x T x F] -> [B x T x H]
        output, _ = self.rnn(x)
        output = self.linear(x)
        output = self.sigmoid(output)
        return output

class Recovery(Module):
    def __init__(self, hidden_size, output_size, num_layers):
        """Latent space -> Original space"""
        self.rnn = GRU(input_size=hidden_size, 
                hidden_size=hidden_size, 
                num_layers=num_layers, 
                batch_first=True)
        self.linear = Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        # [B x T x E] -> [B x T x F]
        output, _ = self.rnn(x)
        output = self.linear(x)
        return output

class Supervisor(Module):
    def __init__(self, input_size, num_layers):
        """Predicts next point"""
        self.rnn = GRU(input_size=input_size, 
                       hidden_size=input_size, 
                       num_layers=num_layers, 
                       batch_first=True)
        self.linear = Linear(in_features=input_size, out_features=input_size)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        # [B x T x E] -> [B x T x E]
        output, _ = self.rnn(x)
        output = self.linear(x)
        output = self.sigmoid(output)
        return output

class TimeGAN(Module):
    def __init__(self, config):
        self.generator = Generator(config["Z_size"], config["hidden_size"], config["num_layers"])
        self.discriminator = Discriminator(config["hidden_size"], config["num_layers"])
        self.embedder = Embedder(config["feature_size"], config["hidden_size"], config["num_layers"])
        self.supervisor = Supervisor(config["hidden_size"], config["num_layers"])
        self.recovery = Recovery(config["hidden_size"], config["feature_size"], config["num_layers"])

        self.embedder_opt = torch.optim.Adam(self.embedder.parameters(), lr=config['lr'])
        self.recovery_opt = torch.optim.Adam(self.recovery.parameters(), lr=config['lr'])
        self.supervisor_opt = torch.optim.Adam(self.supervisor.parameters(), lr=config['lr'])
        self.generator_opt = torch.optim.Adam(self.generator.parameters(), lr=config['lr'])
        self.discriminator_opt = torch.optim.Adam(self.discriminator.parameters(), lr=config['lr'])

        self.config = config

    def recovery_loss(self, x):
        embeddings = self.embedder(x)
        recovery = self.recovery(embeddings)
        next_points = self.supervisor(embeddings)
        next_step_loss = torch.nn.functional.mse_loss(embeddings[:,:-1,:], next_points[:,1:,:])
        recovery_loss = torch.nn.functional.mse_loss(x, recovery)
        sqrt_recovery_loss = 10 * torch.sqrt(recovery_loss)
        full_recovery_loss = sqrt_recovery_loss + 0.1 * next_step_loss
        return full_recovery_loss, sqrt_recovery_loss, recovery_loss

    def supervisor_loss(self, x):
        embeddings = self.embedder(x)
        next_points = self.supervisor(embeddings)
        next_step_loss = torch.nn.functional.mse_loss(x[:,:-1,:], next_points[:,1:,:])
        return next_step_loss

    def discriminator_loss(self, x, z):
        fake_samples = self.generator(z).detach()
        fake_samples_next_step = self.supervisor(fake_samples).detach()
        real_embeddings = self.embedder(x).detach()

        disc_real = self.discriminator(real_embeddings)
        disc_fake_1 = self.discriminator(fake_samples)
        disc_fake_2 = self.discriminator(fake_samples_next_step)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(disc_real, torch.ones_like(disc_real)) + \
               torch.nn.functional.binary_cross_entropy_with_logits(disc_fake_1, torch.zeros_like(disc_fake_1)) + \
               torch.nn.functional.binary_cross_entropy_with_logits(disc_fake_2, torch.zeros_like(disc_fake_2))
        return loss

    def generator_loss(self, x, z):
        fake_samples = self.generator(z)
        fake_samples_next_step = self.supervisor(fake_samples)
        real_samples = self.embedder(x)
        real_samles_next_step = self.supervisor(real_samples)
        fake_recovered = self.recovery(fake_samples_next_step)

        disc_fake_1 = self.discriminator(fake_samples)
        disc_fake_2 = self.discriminator(fake_samples_next_step)

        supervisor_loss = torch.nn.functional.mse_loss(real_samles_next_step[:,:-1,:], real_samples[:,1:,:]) 

        mean_loss = torch.mean(torch.abs((fake_recovered.mean(dim=0)) - (x.mean(dim=0))))
        var_loss = torch.mean(torch.abs(torch.sqrt(fake_recovered.var(dim=0, unbiased=False) + 1e-6) - torch.sqrt(x.var(dim=0, unbiased=False) + 1e-6)))

        loss = torch.nn.functional.binary_cross_entropy_with_logits(disc_fake_1, torch.zeros_like(disc_fake_1)) + \
               torch.nn.functional.binary_cross_entropy_with_logits(disc_fake_2, torch.zeros_like(disc_fake_2)) + \
               100 * torch.sqrt(supervisor_loss) + mean_loss + var_loss
        return loss

    def train_embedder(self, dataloader, n_epochs):
        for epoch in n_epochs:
            mean_loss = 0
            for x in dataloader:
                self.embedder.zero_grad()
                self.recovery.zero_grad()
                _, loss, _ = self.recovery_loss(x)
                loss.backward()
                self.embedder_opt.step()
                self.recovery_opt.step()
                mean_loss += loss
            print(f'Epoch: {epoch}, embedder loss: {mean_loss/len(dataloader)}')

    def train_supervisor(self, dataloader, n_epochs):
        for epoch in n_epochs:
            mean_loss = 0
            for x in dataloader:
                self.supervisor.zero_grad()
                loss = self.supervisor_loss(x)
                loss.backward()
                self.supervisor_opt.step()
                mean_loss += loss
            print(f'Epoch: {epoch}, supervisor loss: {mean_loss/len(dataloader)}')

    def train_gan(self, dataloader, n_epochs):
        for epoch in n_epochs:
            mean_g_loss = 0
            mean_d_loss = 0
            mean_e_loss = 0

            for x in dataloader:
                for _ in range(self.config["G_step"]):
                    z = torch.rand((self.config["batch_size"], self.config["seq_len"], self.config["Z_size"]))

                    self.supervisor.zero_grad()
                    self.recovery.zero_grad()
                    loss = self.generator_loss(x, z)
                    loss.backward()
                    self.supervisor_opt.step()
                    self.generator_opt.step()
                    mean_g_loss += loss

                    self.embedder.zero_grad()
                    self.recovery.zero_grad()
                    loss, _, _ = self.recovery_loss(x)
                    loss.backward()
                    self.embedder_opt.step()
                    self.recovery_opt.step()
                    mean_e_loss += loss

                for _ in range(self.config["D_step"]):
                    z = torch.rand((self.config["batch_size"], self.config["seq_len"], self.config["Z_size"]))

                    self.discriminator.zero_grad()
                    loss = self.discriminator_loss(x, z)
                    mean_d_loss += loss

                    if loss > self.config["disc_loss_thr"]:
                        loss.backward()
                        self.discriminator_opt.step()
                    
            print(f'Epoch: {epoch}, embedder loss: {mean_e_loss/len(dataloader)}')
            print(f'Epoch: {epoch}, generator loss: {mean_g_loss/len(dataloader)}')
            print(f'Epoch: {epoch}, discriminator loss: {mean_d_loss/len(dataloader)}')

    def train(self, dataloader):
        self.train_embedder(dataloader, self.config["n_emb_epochs"])
        self.train_supervisor(dataloader, self.config["n_sup_epochs"])
        self.train_gan(dataloader, self.config["n_gan_epochs"])


config = {
    "n_emb_epochs": 600,
    "n_sup_epochs": 600,
    "n_gan_epochs": 600,
    "disc_loss_thr": 0.15,
    "batch_size": 128,
    "seq_len": 100,
    "feature_size": 28,
    "Z_size": 28,
    "hidden_size": 20,
    "num_layers": 3,
    "lr": 1e-3,
    "G_step": 2,
    "D_step": 1
}