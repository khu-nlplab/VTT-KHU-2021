import math
import torch
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
                           bidirectional=True)

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


class QuestionLevelDifficulty_L(nn.Module):
    def __init__(self):
        super(QuestionLevelDifficulty_L, self).__init__()
        config = RobertaConfig()
        roberta = RobertaModel(config)

        self.roberta = roberta.from_pretrained('roberta-base')

        self.img_encoder = BackboneCNN()
        self.text_encoder = TextEncoder(self.roberta)

        self.logic_class = nn.Linear(768 * 4, 4)
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

        outputs = self.logic_class(uv)
        logic = outputs.max(dim=-1)[1]

        return int((logic + 1).cpu().numpy())


class TransformerAttn(nn.Module):
    def __init__(self):
        super(TransformerAttn, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=2048, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=2048, out_channels=768, kernel_size=1)

    def forward(self, query, key, value):

        scores = torch.matmul(query, key.transpose(1, 0))
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, value)

        context = self.scale_for_low_variance(context)

        output = nn.LeakyReLU()(self.conv1(context.unsqueeze(1).transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)

        output = self.scale_for_low_variance(output)

        return output.squeeze(1)

    def scale_for_low_variance(self, value):
        maximum_value = torch.FloatTensor([math.sqrt(torch.max(value))]).to(self.device)
        if maximum_value > 1.0:
            value.div_(maximum_value)
            return value
        return value
