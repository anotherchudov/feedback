import sys
from torch.utils.checkpoint import checkpoint
sys.path.insert(0, '/home/dlrgy22/Feedback/code/models_training/longformer/sumbission/codes')
import torch
from torch.nn import functional as F
from transformers import DebertaV2Model

class TvmLongformer(torch.nn.Module):
    def __init__(self, grad_checkpt, extra_dense):
        super().__init__()
        self.feats = DebertaV2Model.from_pretrained('microsoft/deberta-v3-large')
        self.feats.pooler = None
        if grad_checkpt:
            self.feats.gradient_checkpointing_enable()
        self.feats.train()

        self.conv1d_layer1 = torch.nn.Conv1d(1024, 1024, kernel_size=1)
        self.conv1d_layer3 = torch.nn.Conv1d(1024, 1024, kernel_size=3, padding=1)
        self.conv1d_layer5 = torch.nn.Conv1d(1024, 1024, kernel_size=5, padding=2)

        if extra_dense:
            self.class_projector = torch.nn.Sequential(
                torch.nn.LayerNorm(1024*3),
                torch.nn.Linear(1024*3, 256),
                torch.nn.GELU(),
                torch.nn.Linear(256, 15)
            )
        else:
            self.class_projector = torch.nn.Sequential(
                torch.nn.LayerNorm(1024*3),
                torch.nn.Linear(1024*3, 15)
            )
    def forward(self, tokens, mask):
        transformer_output = self.feats(tokens, mask, return_dict=False)[0]
        conv_input = transformer_output.transpose(1, 2) # batch, hidden, seq

        conv_output1 = F.relu(self.conv1d_layer1(conv_input)) 
        conv_output3 = F.relu(self.conv1d_layer3(conv_input)) 
        conv_output5 = F.relu(self.conv1d_layer5(conv_input)) 

        concat_output = torch.cat((conv_output1, conv_output3, conv_output5), dim=1).transpose(1, 2)

        output = self.class_projector(concat_output)
        return torch.log_softmax(output, -1)