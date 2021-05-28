import torch
import numpy as np
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig


class BackboneCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = ImageBiLSTM()

    def forward(self, src):

        embedding = self.lstm(src[0].unsqueeze(1))

        return embedding


class ImageBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layers = 1
        self.emb_dim = 128
        self.hid_dim = 384

        self.rnn = nn.LSTM(self.emb_dim, self.hid_dim, self.n_layers,
                           bidirectional=True, dropout=0.1)

        self.pooling = nn.Sequential(nn.Linear(self.hid_dim * 2, self.hid_dim * 2),
                                     nn.Tanh(),
                                     nn.Dropout(p=0.1))

    def forward(self, src):

        _, (hidden, cell) = self.rnn(src)
        bidirectional_hid_1, bidirectional_hid_h2 = hidden[0], hidden[1]
        bidirectional_hid = torch.cat([bidirectional_hid_1, bidirectional_hid_h2], dim=1)

        return self.pooling(bidirectional_hid)


class TextEncoder(nn.Module):
    def __init__(self, roberta):
        super(TextEncoder, self).__init__()
        self.roberta = roberta
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, enc_inputs):

        _, outputs = self.roberta(**enc_inputs)

        return self.dropout(outputs)


class QuestionLevelDifficulty_M(nn.Module):
    def __init__(self):
        super(QuestionLevelDifficulty_M, self).__init__()
        config = RobertaConfig()
        roberta = RobertaModel(config)

        self.roberta = roberta.from_pretrained('roberta-base')

        self.img_encoder = BackboneCNN()
        self.text_encoder = TextEncoder(self.roberta)

        self.memory_class = nn.Linear(768 * 4, 2)

        self.u_attn_layer = nn.ModuleList([TransformerAttn() for _ in range(2)])

    def forward(self,
                text_inputs,
                img_inputs):

        u = self.img_encoder(img_inputs)
        v = self.text_encoder(text_inputs)

        attn_u = v
        attn_v = u

        for layer in self.u_attn_layer:
            attn_u = layer(attn_u, u, u)

        for layer in self.u_attn_layer:
            attn_v = layer(attn_v, v, v)

        uv = torch.cat([u, v], dim=1)
        uv = torch.cat([uv, attn_u], dim=1)
        uv = torch.cat([uv, attn_v], dim=1)

        memory_logits = self.memory_class(uv)
        memory = memory_logits.max(dim=-1)[1]

        return int((memory + 2).cpu().numpy())


class TransformerAttn(nn.Module):
    def __init__(self):
        super(TransformerAttn, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=768, out_channels=2048, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=2048, out_channels=768, kernel_size=1)

    def forward(self, query, key, value):

        scores = torch.matmul(query, key.transpose(1, 0)) / np.sqrt(768)
        attn = nn.Softmax(dim=-1)(scores)

        context = torch.matmul(attn, value)

        output = nn.ReLU()(self.conv1(context.unsqueeze(1).transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2).squeeze(1)

        return output.squeeze(1)