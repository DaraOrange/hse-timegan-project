import torch
from torch.nn import Module, GRU, Linear, Sigmoid

class Generator(Module):
    def __init__(self, z_dim, hidden_size, num_layers):
        """Noise sequence -> Original space"""
        super().__init__()
        # tgan.py:129
        self.e_cell = GRU(
            input_size=z_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear = Linear(
            in_features=hidden_size,
            out_features=hidden_size
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
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.rnn = GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear = Linear(
            in_features=hidden_size,
            out_features=1
        )

    def forward(self, x, T):
        # [B x T x F] -> [B x T x 1]
        H_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=x,
            lengths=T,
            batch_first=True,
            enforce_sorted=False
        )

        H_o, _ = self.rnn(H_packed)
        H_o, _ = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
        )

        logits = self.linear(H_o).squeeze(-1)
        return logits

class Embedder(Module):
    """Original space -> Embedding space"""
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.rnn = GRU(input_size=input_size,
                       hidden_size=hidden_size,
                       num_layers=num_layers,
                       batch_first=True)
        self.linear = Linear(in_features=hidden_size, out_features=hidden_size)
        self.sigmoid = Sigmoid()

    def forward(self, x, T):
        X_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=x,
            lengths=T,
            batch_first=True,
            enforce_sorted=False
        )

        H_o, _ = self.rnn(X_packed)
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
        )

        logits = self.linear(H_o)
        H = self.sigmoid(logits)
        return H


class Recovery(Module):
    def __init__(self, hidden_size, output_size, num_layers):
        """Latent space -> Original space"""
        super().__init__()
        self.rnn = GRU(input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True)
        self.linear = Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x, T):
        H_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=x,
            lengths=T,
            batch_first=True,
            enforce_sorted=False
        )

        H_o, _ = self.rnn(H_packed)
        H_o, _ = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True
        )

        X_tilde = self.linear(H_o)
        return X_tilde


class Supervisor(Module):
    def __init__(self, input_size, num_layers):
        """Predicts next point"""
        super().__init__()
        self.rnn = GRU(input_size=input_size,
                       hidden_size=input_size,
                       num_layers=num_layers,
                       batch_first=True)
        self.linear = Linear(in_features=input_size, out_features=input_size)
        self.sigmoid = Sigmoid()

    def forward(self, x, T):
        H_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=x,
            lengths=T,
            batch_first=True,
            enforce_sorted=False
        )

        H_o, _ = self.rnn(H_packed)
        H_o, _ = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
        )

        logits = self.linear(H_o)
        H_hat = self.sigmoid(logits)
        return H_hat
