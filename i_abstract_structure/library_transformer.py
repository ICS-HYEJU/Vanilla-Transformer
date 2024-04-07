# Use nn.Transforemr in PyTorch
import torch
import torch.nn as nn

# Define Model
transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
src = torch.rand((10,32,512))   # input
tgt = torch.rand((10,32,512))   # output
out = transformer_model(src,tgt)
print(out, out.shape)

import torch.optim as optim

import math
import numpy as np
import random

from matplotlib import pyplot as plt
# ------------------------Create Train Dataset---------------------------------
def generate_random_data(n):
    SOS_token = np.array([2]) # start of sequence
    EOS_token = np.array([3]) # end of sequence
    length = 8

    data = []

    # input -> output
    # 1,1,1,1,1 -> 1,1,1,1,1
    for i in range(n // 3):
        x = np.concatenate((SOS_token, np.ones(length), EOS_token))
        y = np.concatenate((SOS_token, np.ones(length), EOS_token))
        data.append([x, y])

    # 0,0,0,0 -> 0,0,0,0
    for i in range(n // 3):
        x = np.concatenate((SOS_token, np.zeros(length), EOS_token))
        y = np.concatenate((SOS_token, np.zeros(length), EOS_token))
        data.append([x, y])

    # 1,0,1,0 -> 1,0,1,0,1
    for i in range(n // 3):
        x = np.zeros(length)
        start = random.randint(0,1)

        x[start::2] = 1

        y = np.zeros(length)
        if x[-1] == 0:
            y[::2] = 1
        else:
            y[1::2] = 1

        x = np.concatenate((SOS_token, x, EOS_token))
        y = np.concatenate((SOS_token, y, EOS_token))
        data.append([x, y])

    np.random.shuffle(data)

    return data

# make batch size 16
def batchify_data(data, batch_size=16, padding=False, padding_token=-1):
    batches = []
    for idx in range(0, len(data), batch_size):
        # if batch size is not equal to batch_size -> last bit is not obtained
            if idx + batch_size < len(data):
                # load batch max_length, length nomalization by padding
                if padding:
                    max_batch_length = 0
                    # load max batch length
                    for seq in data[idx : idx + batch_size]:
                        if len(seq) > max_batch_length:
                            max_batch_length = len(seq)
                    for seq_idx in range(batch_size):
                        remaining_length = max_batch_length = len(data[idx + seq_idx])
                        data[idx + seq_idx] += [padding_token] * remaining_length

                batches.append(np.array(data[idx : idx + batch_size]).astype(np.int64))
    print(f"{len(batches)} batches of size {batch_size}")

    return batches

# -------------------------------Transformer-----------------------------------

# dim_model : dim of input,output for Endcoder,Decoder (default = 512)
# num_encoder_layers : the number of layers in Encoder (default = 6)
# num_decoder_layers : the number of layers in Decoder (default = 6)
# nhead : the number of multi-head (default = 8)
# dim_feedfoward : dim of FFNN hidden layers (default = 2048)

class Transformer(nn.Module):
    # Constructor
    def __init__(self, num_tokens, dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout_p, ):
        super().__init__()

        # Info
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # Layers
        self.positional_encoder = PositionalEncoding(dim_model=dim_model, dropout_p=dropout_p, max_len=5000)
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.out = nn.Linear(dim_model, num_tokens)

    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # src, Tgt size -> (batch_size, src sequence Length)

        # Embedding + positional encoding = (batch_size, sequence Length, dim_model) is the out size
        ####### Why use '* math.sqrt(self.dim_model)' at embedding of input, output????
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        src = src.permute(1,0,2) # [sequence Length, batch_size, dim_model]
        tgt = tgt.permute(1,0,2)

        # Transformer blocks = (sequence Length, batch_size, num_tokens) is the out size
        transforemr_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.out(transforemr_out)

        return out

    def get_tgt_mask(self,size): # make mask in Decoder
        mask = torch.tril(torch.ones(size, size) == 1) # == 1 : diagonal matrix[size,size]
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # convert zeros to -inf for softmax(q*k) -> 0
        mask = mask.masked_fill(mask == 1, float(0.0)) # convert 1 to 0 ????????????????why????????????????

        return mask

    def create_pad_mask(self, matrix=torch.tensor, pad_token=int): # torch.tensor
        return (matrix == pad_token)


# -----------------------------Positional Encoding------------------------------------
# PE(pos,2i+1) = cos(100002i/dim * pos)
# PE(pos,2i) = sin(100002i/dim * pos)

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model) # dim_model = 512
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1,1) # shape = [??, 1], ex) [1,2,''',6]
        division_term = torch.exp(torch.arange(0, dim_model, 2).float()*(-math.log(10000.0))/dim_model) # /10000^(2i/dim_model)

        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term) # PE(pos,2i)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term) # PE(pos,2i+1)

        # saving buffer (same as parameter without gradients needed)
        # [1, max_len, d_model] -> [max_len, 1, d_model]
        # That is, the shape of pos_encoding is [max_len, 1, d_model].
        # pos_encoding[0] means a position encoding vector for the first word, and thus
        # pos_encoding[20] is a position encoding vector for the 21-th word.
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0,1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor): # torch.tensor
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


# ---------------------------------Training-------------------------------
def train_loop(model, opt, loss_fn, dataloader):
    model.train()
    total_loss = 0

    for batch in dataloader:
        x, y = batch[:, 0], batch[:, 1]
        x, y = torch.tensor(x).to(device), torch.tensor(y).to(device)

        # tgt shifted by 1 and predict the token at position. because use <sos> at index0.
        y_input = y[:, :-1]
        y_expected = y[:, 1:]

        # load mask to masking next word
        sequence_length = y_input.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)

        # send x, y_input and tgt_mask to model, training
        pred = model(x, y_input, tgt_mask)

        # permute -> batch first
        pred = pred.permute(1, 2, 0)
        loss = loss_fn(pred, y_expected)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.detach().item()

    return total_loss / len(dataloader)

# -----------------------------Validataion----------------------------
def validation_loop(model, loss_fn, dataloader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch[:, 0], batch[:, 1]
            x, y = torch.tensor(x, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)

            y_input = y[:,:-1]
            y_expected = y[:,1:]

            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            pred = model(x, y_input, tgt_mask)

            pred = pred.permute(1, 2, 0)
            loss = loss_fn(pred, y_expected)
            total_loss += loss.detach().item()

    return total_loss / len(dataloader)

# -----------------------------Train and Validate--------------------------------------
def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs):
    # generate list for plottiong
    train_loss_list, validation_loss_list = [], []

    print("Training and validationg model")
    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch + 1}", "-"*25)

        train_loss = train_loop(model, opt, loss_fn, train_dataloader)
        train_loss_list += [train_loss]

        validation_loss = validation_loop(model, loss_fn, val_dataloader)
        validation_loss_list += [validation_loss]

        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print()

    return train_loss_list, validation_loss_list

# ------------------------------Inference----------------------------------
# make predict function and Test
def predict(model, input_sequence, max_length=15, SOS_token=2, EOS_token=3):
    model.eval()

    y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)

    num_tokens = len(input_sequence[0])

    for _ in range(max_length):
        # Get source mask
        tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)

        pred = model(input_sequence, y_input, tgt_mask)

        next_item = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
        next_item = torch.tensor([[next_item]], device=device)

        # Concatenate previous input with predicted best word
        y_input = torch.cat((y_input, next_item), dim=1)


        # Stop if model predicts end of sentence
        if next_item.view(-1).item() == EOS_token:
            break

    return y_input.view(-1).tolist()





if __name__ == '__main__':
    # Create Dataset
    train_data = generate_random_data(9000)
    val_data = generate_random_data(3000)
    # Load Dataloader(load dataset
    train_dataloader = batchify_data(train_data)
    val_dataloader = batchify_data(val_data)
    # Define
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer(num_tokens=4,
                        dim_model=8,
                        num_heads=2,
                        num_encoder_layers=3,
                        num_decoder_layers=3,
                        dropout_p=0.1).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    # Train
    train_loss_list, validation_loss_list = fit(model, opt, loss_fn, train_dataloader, val_dataloader, 10)
    # Print plot
    plt.plot(train_loss_list, label='Train loss')
    plt.plot(validation_loss_list, label = 'Validation loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.show()


    # Inference Example
    # Here we test some examples to observe how the model predicts
    examples = [
        torch.tensor([[2, 0, 0, 0, 0, 0, 0, 0, 0, 3]], dtype=torch.long, device=device),
        torch.tensor([[2, 1, 1, 1, 1, 1, 1, 1, 1, 3]], dtype=torch.long, device=device),
        torch.tensor([[2, 1, 0, 1, 0, 1, 0, 1, 0, 3]], dtype=torch.long, device=device),
        torch.tensor([[2, 0, 1, 0, 1, 0, 1, 0, 1, 3]], dtype=torch.long, device=device),
        torch.tensor([[2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 3]], dtype=torch.long, device=device),
        torch.tensor([[2, 0, 1, 3]], dtype=torch.long, device=device)
    ]

    for idx, example in enumerate(examples):
        result = predict(model, example)
        print(f"Example {idx}")
        print(f"Input: {example.view(-1).tolist()[1:-1]}")
        print(f"Continuation: {result[1:-1]}")
        print()