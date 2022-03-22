import torch
from torch.nn import Module, GRU, Linear, Sigmoid

class Generator(Module):
    def __init__(self, z_dim, hidden_dim, num_layers):
        """Noise sequence -> Original space"""
        super().__init__()
        # tgan.py:129
        self.e_cell = GRU(
            input_size=z_dim, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.linear = Linear(
            in_features=hidden_dim, 
            out_features=hidden_dim
        )
        self.sigmoid = Sigmoid()

    def forward(self, Z, T):
        # [B x T x Z] -> [B x T x H]
        # Dynamic RNN input for ignoring paddings
        
        Z = torch.nn.utils.rnn.pack_padded_sequence(
            input=Z, 
            lengths=T, 
            batch_first=True, 
            enforce_sorted=False
        )

        # tgan.py:131
        e_outputs, e_last_states = self.e_cell(Z)
        
        # tgan.py:133
        e_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=e_outputs, 
            batch_first=True
        )
        E = self.sigmoid(self.linear(e_outputs))

        return E

class Discriminator(Module):
    """Original space -> Logits"""
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.e_cell = GRU(
            input_size=hidden_dim, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.linear = Linear(
            in_features=hidden_dim, 
            out_features=1
        )

    def forward(self, x):
        # [B x T x F] -> [B x T x 1]
        pass

class Embedder(Module):
    """Original space -> Embedding space"""
    def __init__(self, input_size, hidden_dim, num_layers):
        super().__init__()
        self.rnn = GRU(input_size=input_size, 
                       hidden_dim=hidden_dim, 
                       num_layers=num_layers, 
                       batch_first=True)
        self.linear = Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        # [B x T x F] -> [B x T x H]
        pass
class Recovery(Module):
    def __init__(self, hidden_dim, output_size, num_layers):
        """Latent space -> Original space"""
        super().__init__()
        self.rnn = GRU(input_size=hidden_dim, 
                hidden_dim=hidden_dim, 
                num_layers=num_layers, 
                batch_first=True)
        self.linear = Linear(in_features=hidden_dim, out_features=output_size)

    def forward(self, x):
        # [B x T x E] -> [B x T x F]
        pass

class Supervisor(Module):
    def __init__(self, input_size, num_layers):
        """Predicts next point"""
        super().__init__()
        self.rnn = GRU(input_size=input_size, 
                       hidden_dim=input_size, 
                       num_layers=num_layers, 
                       batch_first=True)
        self.linear = Linear(in_features=input_size, out_features=input_size)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        # [B x T x E] -> [B x T x E]
        pass
