import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat

import math


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, seq_len):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        # Encoding - From formula
        pos_encoding = torch.zeros(seq_len, hidden_dim)
        positions_list = torch.arange(0, seq_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0)) / hidden_dim)

        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)  # shape = [60,128] = [seq_len, hidden_dim]
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # saving buffer (same as parameter without gradients needed)
        # [seq_len, hidden_dim] -> [seq_len, 1, hidden_dim]
        # That is, the shape of pos_encoding is [seq_len, 1, hidden_dim].
        # pos_encoding[0] means a position encoding vector for the first word, and thus
        # pos_encoding[20] is a position encoding vector for the 21-th word.

        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Input Embedding + pos encoding
        return token_embedding + self.pos_encoding


class Multiheadattention(nn.Module):
    def __init__(self, hidden_dim: int, num_head: int, device=torch.device('cpu')):
        super().__init__()
        # Size of embedding_dim and d_model in paper = 512
        self.hidden_dim = hidden_dim  # this is d_model
        # Number of heads in paper = 8
        self.num_head = num_head
        # Value of head_dim, d_key, d_query, and d_value in paper is 64
        # , which is obtained by (512/8)
        assert (hidden_dim % num_head) == 0, "(hidden_dim % num_head) is not zero"
        self.head_dim = hidden_dim // num_head
        # This is a sqrt function for normalization.
        self.scale = self.head_dim ** -0.5

        # nn.Linear(in_features, out_features) : Performing a linear transformation of the data.
        # "Performing a linear transformation of the data" can be represented symbolically as: y = wx + b
        # When creating an nn.Linear object, the weight matrix (size: [out_features * in_features]) and the bias vector (size: [out_features]) are initialized randomly.

        # Q = q * w, K = k * 2, V = v * w -> Use  Linear and Create Q, K, V matrix
        # Q matrix
        self.fcQ = nn.Linear(hidden_dim, hidden_dim)
        # K matrix
        self.fcK = nn.Linear(hidden_dim, hidden_dim)
        # V matrix
        self.fcV = nn.Linear(hidden_dim, hidden_dim)
        #
        self.fcOut = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, srcQ, srcK, srcV, mask=None):
        # ----- Scaled Dot Production Attention -----#

        # Input shape : (bs, seq_len, hidden_dim), bs = batch_size
        # Assumption: (seq_len=128), (hidden_dim=512)
        #                  |                 | -> Each word is embedded, the shape of which is (512)
        #                  |--------------------> One_batch includes 128 words.
        # Therefore, the shape of one batch is (128, 512)
        Q = self.fcQ(srcQ)
        K = self.fcK(srcK)
        V = self.fcV(srcV)

        # Implementing multi-head attention using the num_head parameter.
        # The num_head parameter is placed at index 1, and for each head, there is an operation with dimensions (seq_len, head_dim), indicating that operations are performed separately for each head.
        # hidden_dim = num_head(8) * head_dim(64)
        Q = rearrange(
            Q,
            'bs seq_len (num_head head_dim) -> bs num_head seq_len head_dim',
            num_head=self.num_head
        )
        K_T = rearrange(
            K,
            'bs seq_len (num_head head_dim) -> bs num_head head_dim seq_len',
            num_head=self.num_head
        )
        V = rearrange(
            V,
            'bs seq_len (num_head head_dim) -> bs num_head seq_len head_dim',
            num_head=self.num_head
        )
        attention_energy = torch.matmul(Q, K_T) * self.scale
        # attetnion_energy shape : (bs, num_head, seq_len, seq_len)

        if mask is not None:
            attention_energy: torch.masked_fill(attention_energy, (mask == 0), dim=-1)

        attention_energy = torch.softmax(attention_energy, dim=-1)

        # softmax (Q * K_T) * V
        result = torch.matmul(attention_energy, V)
        # result shape : (bs, num_head, seq_len, head_dim)

        # --- End of Scaled Dot Product Attention ---#

        # Concatenate
        result = rearrange(
            result,
            'bs num_head seq_len head_dim -> bs seq_len (num_head head_dim)'
        )

        # Linear
        result = self.fcOut(result)

        return result


class FFN(nn.Module):
    def __init__(self, hidden_dim: int, inner_dim: int):
        super().__init__()

        # hidden_dim (or d_model) = 512 in paper
        self.hidden_dim = hidden_dim
        # inner_dim = 2048 in paper
        self.inner_dim = inner_dim

        self.fc1 = nn.Linear(hidden_dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, input):
        output = input  # input.shape = (bs, num_head, hidden_dim)
        output = self.fc1(output)  # output.shape = (bs, num_head, inner_dim)
        output = self.relu(output)  # output.shape = (bs, num_head, inner_dim)
        output = self.fc2(output)  # output.shape = (bs, num_head, hidden_dim)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, seq_len: int, hidden_dim: int, num_head: int, inner_dim: int, device=torch.device('cpu')):
        super().__init__()

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.inner_dim = inner_dim

        self.multiheadattention = Multiheadattention(hidden_dim, num_head, device)
        self.ffn = FFN(hidden_dim, inner_dim)
        self.layerNorm = nn.LayerNorm(hidden_dim)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

    def forward(self, input, mask=None):
        # input shape : (bs, seq_len, hidden_dim)
        # hidden_dim is the size of a embedded vector

        # Encoder attention
        output = self.multiheadattention(srcQ=input, srcK=input, srcV=input, mask=mask)
        output = self.dropout1(output)
        output = input + output
        output = self.layerNorm(output)

        output_ = self.ffn(output)
        output_ = self.dropout2(output_)
        output = output + output_
        output = self.layerNorm(output)

        # output shape : (bs, seq_len, hidden_dim)
        return output


class Encoder(nn.Module):
    def __init__(self, seq_len: int, N: int, hidden_dim: int, num_head: int, inner_dim: int, vocab_size: int):
        super().__init__()

        # N : number of encoder layer repeated
        self.N = N
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.inner_dim = inner_dim

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim, padding_idx=0)
        self.position_embedding = PositionalEncoding(hidden_dim, seq_len)
        self.enc_layers = nn.ModuleList(EncoderLayer(seq_len, hidden_dim, num_head, inner_dim) for _ in range(self.N))

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input, verbose=False):
        # input : (bs, seq_len)
        if verbose:
            print(f'Input shape = {input.shape}')
        mask = makeMask(input, option='padding')

        # embedding layer
        output = self.embedding(input)
        # output.shape : (bs, seq_len, hidden_dim)

        # PositionalEncoding -> return [self.embedding(input) + pos_encode]
        output = self.position_embedding(output)

        # Dropout
        output = self.dropout(output)

        # N encoder layer
        for layer in self.enc_layers:
            output = layer(output, mask)  # shape : bs, seq_len, hidden_dim

        return output


def makeMask(tensor, option: str, PAD_IDX: int = 0, device='cpu') -> torch.Tensor:
    '''
    tensor shape (bs, seq_len)
    '''
    if option == 'padding':
        tmp = torch.full_like(tensor, fill_value=PAD_IDX).to(device)
        # -> tmp shape : (bs, seq_len)
        mask = (tensor != tmp).float()
        # -> mask shape : (bs, seq_len)
        mask = rearrange(mask, 'bs seq_len -> bs 1 1 seq_len')
        # -> mask shape : (bs, 1, seq_len, seq_len)


    elif option == 'lookahead':
        # let len(seq_len of srcQ) == len(seq_len of srcK)
        # tensor shape : (bs, seq_len)
        padding_mask = makeMask(tensor, 'padding')
        # -> padding_mask shape : (bs, 1, seq_len, seq_len)
        # -> padding_mask.shape[3] = seq_len
        padding_mask = repeat(
            padding_mask, 'bs 1 1 k_len -> bs 1 new k_len', new=padding_mask.shape[3]
        )
        # padding_mask : (bs 1 seq_len seq_len)

        '''
        Example of padding_mask:
          Input: tensor = [[1., 1., 1., 1., 0., 0., 0., 0.]]     # (bs, seq_len)
          Output: padding_mask = tensor([[                       # (bs, 1, seq_len, seq_len)
                          [1., 1., 1., 1., 0., 0., 0., 0.]
                          [1., 1., 1., 1., 0., 0., 0., 0.]
                          [1., 1., 1., 1., 0., 0., 0., 0.]
                          [1., 1., 1., 1., 0., 0., 0., 0.]
                          [1., 1., 1., 1., 0., 0., 0., 0.]
                          [1., 1., 1., 1., 0., 0., 0., 0.]
                          [1., 1., 1., 1., 0., 0., 0., 0.]
                          [1., 1., 1., 1., 0., 0., 0., 0.]
                          ]])
        '''
        mask = torch.ones_like(padding_mask)
        mask = torch.tril(mask)
        '''
        Example of 'mask':
          Result of torch.ones_like(padding_mask):
            mask = tensor([[
                          [1., 1., 1., 1., 1., 1., 1., 1.]
                          [1., 1., 1., 1., 1., 1., 1., 1.]
                          [1., 1., 1., 1., 1., 1., 1., 1.]
                          [1., 1., 1., 1., 1., 1., 1., 1.]
                          [1., 1., 1., 1., 1., 1., 1., 1.]
                          [1., 1., 1., 1., 1., 1., 1., 1.]
                          [1., 1., 1., 1., 1., 1., 1., 1.]
                          [1., 1., 1., 1., 1., 1., 1., 1.]
                          ]])
          Result of torch.tril(mask):
            mask = tensor([[
                          [1., 0., 0., 0., 0., 0., 0., 0.],
                          [1., 1., 0., 0., 0., 0., 0., 0.],
                          [1., 1., 1., 0., 0., 0., 0., 0.],
                          [1., 1., 1., 1., 0., 0., 0., 0.],
                          [1., 1., 1., 1., 1., 0., 0., 0.],
                          [1., 1., 1., 1., 1., 1., 0., 0.],
                          [1., 1., 1., 1., 1., 1., 1., 0.],
                          [1., 1., 1., 1., 1., 1., 1., 1.]
                          ]])
        '''
        mask = mask * padding_mask
        # ic(mask.shape)

        '''
        Example
        tensor([[
         [1., 0., 0., 0., 0., 0., 0., 0.],
         [1., 1., 0., 0., 0., 0., 0., 0.],
         [1., 1., 1., 0., 0., 0., 0., 0.],
         [1., 1., 1., 1., 0., 0., 0., 0.],
         [1., 1., 1., 1., 0., 0., 0., 0.],
         [1., 1., 1., 1., 0., 0., 0., 0.],
         [1., 1., 1., 1., 0., 0., 0., 0.],
         [1., 1., 1., 1., 0., 0., 0., 0.]]])
        '''

    return mask


class DecoderLayer(nn.Module):
    def __init__(self, seq_len, hidden_dim, num_head, inner_dim):
        super().__init__()

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.inner_dim = inner_dim

        self.multiheadattention1 = Multiheadattention(hidden_dim, num_head)
        self.layerNorm1 = nn.LayerNorm(hidden_dim)
        self.multiheadattention2 = Multiheadattention(hidden_dim, num_head)
        self.layerNorm2 = nn.LayerNorm(hidden_dim)
        self.ffn = FFN(hidden_dim, inner_dim)
        self.layerNorm3 = nn.LayerNorm(hidden_dim)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)

    def forward(self, input, enc_output, paddingMask, lookaheadMask):
        # Decoder Input = Encoder Output
        # input.shape = (bs, seq_len, hidden_dim)

        # first multiheadattention (Use masked self-attention : lookahead)
        output = self.multiheadattention1(input, input, input, lookaheadMask)
        output = self.dropout1(output)
        output = output + input
        output = self.layerNorm1(output)

        # second multiheadattention (Use Encoder-Decoder attention : padding)
        output_ = self.multiheadattention2(output, enc_output, enc_output, paddingMask)
        output_ = self.dropout2(output_)
        output = output_ + output
        output = self.layerNorm2(output)

        # Feedforward Network
        output_ = self.ffn(output)
        output_ = self.dropout3(output_)
        output = output + output_
        output = self.layerNorm3(output)

        return output


class Decoder(nn.Module):
    def __init__(self, seq_len, N, hidden_dim, num_head, inner_dim, vocab_size: int):
        super().__init__()

        # N : the number of encoder layer repeated
        self.N = N
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.inner_dim = inner_dim

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim, padding_idx=0)
        self.position_embedding = PositionalEncoding(hidden_dim, seq_len)

        self.dec_layers = nn.ModuleList(DecoderLayer(seq_len, hidden_dim, num_head, inner_dim) for _ in range(self.N))

        self.dropout = nn.Dropout(p=0.1)

        self.finalFc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input, enc_src, enc_output):
        # input.shape = (bs, seq_len)
        # enc_src.shape = (bs, seq_len)
        # enc_output.shape = (bs, seq_len, hidden_dim)

        lookaheadMask = makeMask(input, option='lookahead')
        paddingMask = makeMask(enc_src, option='padding')

        # embedding layer
        output = self.embedding(input)

        # Positional Embedding
        output = self.position_embedding(output)

        # DropOut
        output = self.dropout(output)

        # N decoder layer
        for layer in self.dec_layers:
            output = layer(output, enc_output, paddingMask, lookaheadMask)
            # output.shape = (bs, seq_len, hidden_dim)

        logits = self.finalFc(output)
        # logits.shape = (bs, seq_len, vocab_size)

        output = torch.softmax(logits, dim=-1)

        output = torch.argmax(output, dim=-1)
        # output.shape = (bs, seq_len)

        return output, logits


class TransformerModel(nn.Module):
    def __init__(self, seq_len, vocab_size, N=6, hidden_dim=512, num_head=8, inner_dim=2048):
        super().__init__()
        self.encoder = Encoder(seq_len, N, hidden_dim=hidden_dim, num_head=num_head, inner_dim=inner_dim,
                               vocab_size=vocab_size)
        self.decoder = Decoder(seq_len, N, hidden_dim=hidden_dim, num_head=num_head, inner_dim=inner_dim,
                               vocab_size=vocab_size)

    def forward(self, enc_src, dec_src):
        # enc_src.shape = (bs, seq_len)
        # dec_src.shape = (bs, seq_len)

        enc_output = self.encoder(enc_src)
        output, logits = self.decoder(input=dec_src, enc_src=enc_src, enc_output=enc_output.detach())
        # logits.shape = (bs, seq_len, vocab_size)

        return output, logits


if __name__ == '__main__':
    from i_abstract_structure.config.config import get_config_dict
    from i_abstract_structure.dataset.dataset import bible_dataset as bible

    cfg = get_config_dict()

    model = TransformerModel(seq_len=cfg['dataset_info']['seq_len'], vocab_size=cfg['dataset_info']['vocab_size'],
                             N=cfg['dataset_info']['N'], hidden_dim=cfg['dataset_info']['hidden_dim'],
                             num_head=cfg['dataset_info']['num_head'], inner_dim=cfg['dataset_info']['inner_dim'])

    bible_object = bible(cfg=cfg, seq_len=cfg['dataset_info']['seq_len'])
    bible_dataloader = torch.utils.data.DataLoader(bible_object, batch_size=cfg['dataset_info']['batch_size'], drop_last=True)

    for idx, data in enumerate(bible_dataloader):
        src = data[0]
        trg_input = data[1]
        trg_output = data[2]

        output, logits = model(enc_src=src, dec_src=trg_input)