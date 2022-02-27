import sys
from torch.utils.checkpoint import checkpoint
sys.path.insert(0, '../codes')
import torch
from torch.nn import functional as F
from transformers import DebertaV2Model

from torch.nn.parallel import DistributedDataParallel as DDP

class DebertaV3Large(torch.nn.Module):
    """microsoft/deberta-v3-large"""
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.feats = DebertaV2Model.from_pretrained('microsoft/deberta-v3-large')
        self.feats.pooler = None

        if args.grad_checkpt:
            self.feats.gradient_checkpointing_enable()

        if args.cnn1d:
            self.conv1d_layer1 = torch.nn.Conv1d(1024, 1024, kernel_size=1)
            self.conv1d_layer3 = torch.nn.Conv1d(1024, 1024, kernel_size=3, padding=1)
            self.conv1d_layer5 = torch.nn.Conv1d(1024, 1024, kernel_size=5, padding=2)

            self.output_length = 1024 * 3
        else:
            self.output_length = 1024

        if args.extra_dense:
            self.class_projector = torch.nn.Sequential(
                torch.nn.LayerNorm(self.output_length),
                torch.nn.Linear(self.output_length, 256),
                torch.nn.GELU(),
                torch.nn.Linear(256, 15)
            )
        else:
            self.class_projector = torch.nn.Sequential(
                torch.nn.LayerNorm(self.output_length),
                torch.nn.Linear(self.output_length, 15)
            )

    def forward(self, tokens, mask):
        transformer_output = self.feats(tokens, mask, return_dict=False)[0]
        
        if self.args.cnn1d:
            conv_input = transformer_output.transpose(1, 2) # batch, hidden, seq

            conv_output1 = F.relu(self.conv1d_layer1(conv_input)) 
            conv_output3 = F.relu(self.conv1d_layer3(conv_input)) 
            conv_output5 = F.relu(self.conv1d_layer5(conv_input)) 

            concat_output = torch.cat((conv_output1, conv_output3, conv_output5), dim=1).transpose(1, 2)
            output = self.class_projector(concat_output)
        else:
            output = self.class_projector(transformer_output) # batch, seq, hidden

        return torch.log_softmax(output, -1)


def get_model(args):
    if args.model == 'microsoft/deberta-v3-large':
        model = DebertaV3Large(args).to(args.device)

        # dropout layer
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = args.dropout_ratio

    # distributed training
    if args.ddp:
        model = DDP(model, device_ids=[args.rank], output_device=args.rank)
        model.to(args.device)

    return model