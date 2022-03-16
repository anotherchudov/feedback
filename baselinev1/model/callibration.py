
import torch
from torch import nn

class Callibration(nn.Module):
    """Used for callibrating the token category sequence
    
    before : 0 1 2 2 0 3 2
    after  : 0 1 2 2 2 2 2
    """
    def __init__(self,
                 hidden_size=128,
                 n_layers=2,
                 device='cuda:0'):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device

        self.embedding = nn.Embedding(16, hidden_size, padding_idx=15)
        self.lstm = nn.LSTM(hidden_size,
                            hidden_size,
                            n_layers,
                            bidirectional=True,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, 16)
        
    def init_hidden(self, batch_size):
        h = torch.zeros(self.n_layers * 2, batch_size, self.hidden_size)
        h = h.to(self.device)

        c = torch.zeros(self.n_layers * 2, batch_size, self.hidden_size)
        c = c.to(self.device)

        return (h, c)

    def forward(self, x):
        batch_size, seq_len = x.size()
        out = self.embedding(x)
        
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(out, hidden)
        
        out = out.contiguous().view(batch_size, -1, self.hidden_size * 2)
        out = self.fc(out)
        
        return out