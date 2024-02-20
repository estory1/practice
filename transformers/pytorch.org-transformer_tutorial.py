#%%[markdown]
## Using Pytorch Transformers module
#
# Date created: 20230920
#
# Src: [Language Modeling with nn.Transformer and torchtext — PyTorch Tutorials 2.0.1+cu117 documentation](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

#%%
import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

#%%
# import my lib for device selection
import sys
import os
sys.path.append( os.path.expanduser('~') + "/home/dev/dev/libs_mine" )
import torch_device

device = torch_device.get_best_device()
# device = torch.device("cpu")


#%%[markdown]
# Example TEL usage: https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html#torch.nn.TransformerEncoderLayer
# 
# Tensor structure when `batch_first=True` is:
#   Outer []: batches
#   Middle []: samples in batch
#   Inner []: feature values in sample
# ```
# >>> encoder_layer = nn.TransformerEncoderLayer(d_model=8, nhead=8, batch_first=True)
# >>> src = torch.rand(2, 10, 8)
# >>> out = encoder_layer(src)
# >>> out
# tensor([[[-1.5788e-01, -7.7597e-01,  1.1261e+00, -1.5683e+00,  1.6315e+00,
#            7.0163e-01, -3.4935e-01, -6.0764e-01],
#          [ 1.8114e+00, -2.0576e+00,  5.3989e-01, -2.8529e-01,  3.1905e-02,
#            2.1563e-01, -2.5404e-01, -1.8926e-03],
#          [ 1.7758e+00, -1.4137e+00,  7.2128e-01, -1.2825e+00,  3.1556e-01,
#            2.7866e-01, -6.5995e-01,  2.6490e-01],
#          [ 1.5085e+00, -6.4184e-01,  8.3536e-01, -6.3307e-01, -1.6667e+00,
#            6.9277e-02,  1.0678e+00, -5.3941e-01],
#          [ 1.1057e+00, -1.7044e+00,  1.2267e+00, -1.8348e-01,  6.1843e-01,
#           -8.4744e-01,  6.7003e-01, -8.8563e-01],
#          [ 1.5493e+00, -1.1069e+00,  5.9067e-01, -1.5550e+00,  6.2200e-01,
#           -3.5651e-01,  8.5639e-01, -6.0004e-01],
#          [ 1.1541e+00,  8.0896e-01, -4.8473e-02,  9.2812e-01, -9.1226e-01,
#           -8.5991e-01,  6.9055e-01, -1.7611e+00],
#          [-4.7557e-01, -5.4610e-01,  2.0213e+00,  2.5855e-01, -9.8594e-01,
#           -1.7362e-01,  1.0268e+00, -1.1254e+00],
#          [ 1.9909e+00, -1.5440e+00,  2.9410e-01, -7.1755e-01,  4.5857e-01,
#           -8.6833e-01,  1.1670e-01,  2.6960e-01],
#          [ 1.0702e+00, -1.6596e+00,  1.1369e+00, -1.4442e+00,  6.5805e-01,
#           -3.2933e-01,  1.8557e-01,  3.8239e-01]],

#         [[ 2.3308e+00, -1.1455e+00, -4.1205e-02,  5.6604e-01, -4.7572e-01,
#           -7.7549e-01, -2.2846e-01, -2.3047e-01],
#          [ 7.9482e-01, -1.8893e+00,  3.5672e-01, -3.9444e-01,  4.9138e-02,
#            7.6432e-01,  1.3591e+00, -1.0403e+00],
#          [ 1.5597e+00, -1.4323e+00,  1.2754e+00, -7.4382e-01, -1.0862e+00,
#            3.6613e-01,  1.3095e-01, -6.9887e-02],
#          [ 1.2245e+00, -1.0506e+00,  1.5076e+00, -1.4003e+00, -1.1746e-01,
#           -9.2221e-01,  3.0445e-01,  4.5395e-01],
#          [ 1.0699e+00, -1.9875e+00,  8.9205e-01, -1.2363e+00,  4.7205e-01,
#            2.5546e-01,  5.4109e-01, -6.6580e-03],
#          [ 1.3873e+00, -1.4802e+00,  1.1516e+00, -8.7333e-01, -3.8229e-01,
#           -7.4303e-02,  1.0320e+00, -7.6084e-01],
#          [ 1.8370e+00, -5.0212e-01,  7.6123e-01, -1.2952e+00,  9.8337e-01,
#           -4.1503e-01, -5.4374e-01, -8.2550e-01],
#          [ 1.2804e+00, -1.8022e+00,  8.7449e-01, -1.4225e+00,  1.7135e-01,
#            1.7489e-01,  3.2609e-01,  3.9741e-01],
#          [ 6.2777e-02, -1.4451e+00, -2.3473e-01,  1.1170e+00, -7.9930e-01,
#            4.3427e-01,  1.7370e+00, -8.7192e-01],
#          [ 1.1838e+00, -2.2432e+00, -3.8276e-01,  3.3566e-01,  1.0236e+00,
#            2.9505e-01, -3.8027e-01,  1.6816e-01]]],
#        grad_fn=<NativeLayerNormBackward0>)
# >>> src
# tensor([[[0.3721, 0.7492, 0.8619, 0.0772, 0.8820, 0.9719, 0.3205, 0.2775],
#          [0.7480, 0.1302, 0.3466, 0.4326, 0.4863, 0.6283, 0.0286, 0.3590],
#          [0.7290, 0.5189, 0.3270, 0.1474, 0.4671, 0.6596, 0.1022, 0.5369],
#          [0.7212, 0.3065, 0.6444, 0.6304, 0.0596, 0.6528, 0.6921, 0.2936],
#          [0.3859, 0.6039, 0.9116, 0.9896, 0.9044, 0.4249, 0.7000, 0.4843],
#          [0.9455, 0.6426, 0.5382, 0.0807, 0.9376, 0.5500, 0.9242, 0.4611],
#          [0.3272, 0.8435, 0.3523, 0.9952, 0.2817, 0.2324, 0.3142, 0.0992],
#          [0.2466, 0.9334, 0.8004, 0.6900, 0.1778, 0.4512, 0.4968, 0.3571],
#          [0.8291, 0.3604, 0.2715, 0.4255, 0.7776, 0.1254, 0.2600, 0.8059],
#          [0.4766, 0.3675, 0.8219, 0.3420, 0.7989, 0.5035, 0.7001, 0.7767]],

#         [[0.8448, 0.6641, 0.1681, 0.9251, 0.4160, 0.3496, 0.1650, 0.4270],
#          [0.7849, 0.4124, 0.1875, 0.5633, 0.4887, 0.8903, 0.7781, 0.1462],
#          [0.6376, 0.4862, 0.6047, 0.4060, 0.0415, 0.7630, 0.2638, 0.3691],
#          [0.2399, 0.3392, 0.7722, 0.1301, 0.2304, 0.0674, 0.2556, 0.6389],
#          [0.4644, 0.4924, 0.5676, 0.3577, 0.5273, 0.6959, 0.8188, 0.6229],
#          [0.5251, 0.0398, 0.8627, 0.5133, 0.4192, 0.5202, 0.5631, 0.3803],
#          [0.9040, 0.9287, 0.5221, 0.1857, 0.9609, 0.5506, 0.0679, 0.2287],
#          [0.4471, 0.8022, 0.6265, 0.4609, 0.5776, 0.8329, 0.6593, 0.8886],
#          [0.4792, 0.4389, 0.0386, 0.9353, 0.0246, 0.6307, 0.9659, 0.2416],
#          [0.7047, 0.0855, 0.0391, 0.6931, 0.8234, 0.9508, 0.2143, 0.8368]]])
# ```

#%%
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

#%%
class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output

#%%
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

#%%
def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

# ``train_iter`` was "consumed" by the process of building the vocab,
# so we have to create it again
train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
%%time
def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into ``bsz`` separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape ``[N]``
        bsz: int, batch size

    Returns:
        Tensor of shape ``[N // bsz, bsz]``
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size)  # shape ``[seq_len, batch_size]``
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)

#%%
bptt = 35
def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape ``[full_seq_len, batch_size]``
        i: int

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
        target has shape ``[seq_len * batch_size]``
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

#%%
%%time

ntokens = len(vocab)  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 2  # number of heads in ``nn.MultiheadAttention``
dropout = 0.2  # dropout probability
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

#%%
%%time

import time

criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

def train(model: nn.Module) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        output = model(data)
        output_flat = output.view(-1, ntokens)
        loss = criterion(output_flat, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            seq_len = data.size(0)
            output = model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)

#%%
%%time

best_val_loss = float('inf')
epochs = 3

with TemporaryDirectory() as tempdir:
    best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(model)
        val_loss = evaluate(model, val_data)
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_params_path)

        scheduler.step()
    model.load_state_dict(torch.load(best_model_params_path)) # load best model states

#%%
%%time

test_loss = evaluate(model, test_data)
test_ppl = math.exp(test_loss)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | '
      f'test ppl {test_ppl:8.2f}')
print('=' * 89)

# %%[markdown]
### Understanding the Perplexity metric a bit, differing from these descriptions:
# - https://huggingface.co/docs/transformers/perplexity
# - https://en.wikipedia.org/wiki/Perplexity
#
# ```
# >>> import math
# >>> total_loss = 0 ; log_interval = 200 ;  cur_loss = total_loss / log_interval ;   math.exp(cur_loss)
# 1.0
# >>> total_loss = 0.5 ; log_interval = 200 ;  cur_loss = total_loss / log_interval ;   math.exp(cur_loss)
# 1.0025031276057952
# >>> total_loss = 1 ; log_interval = 200 ;  cur_loss = total_loss / log_interval ;   math.exp(cur_loss)
# 1.005012520859401
# >>> total_loss = 10 ; log_interval = 200 ;  cur_loss = total_loss / log_interval ;   math.exp(cur_loss)
# 1.0512710963760241
# >>> total_loss = 100 ; log_interval = 200 ;  cur_loss = total_loss / log_interval ;   math.exp(cur_loss)
# 1.6487212707001282
# >>> total_loss = 1000 ; log_interval = 200 ;  cur_loss = total_loss / log_interval ;   math.exp(cur_loss)
# 148.4131591025766
# >>> total_loss = 10000 ; log_interval = 200 ;  cur_loss = total_loss / log_interval ;   math.exp(cur_loss)
# 5.184705528587072e+21
# ```



#%%[markdown]
## That's Pytorch's Transformer implementation. What about writing my own?
#
### Below, attempt to implement Vaswani, et al's 2017 "Attention Is All You Need" transformer architecture using Pytorch.

#%%
logging = True
def scaled_dot_product_attention( Q: torch.Tensor , K: torch.Tensor, V: torch.Tensor, dev: torch.device ) -> torch.Tensor:
    """3.2.1 Scaled Dot-Product Attention"""
    if logging: print(f"[SDPA] Q.shape={Q.shape}, K.shape={K.shape}, V.shape={V.shape} ; \n\tQ={Q}, \n\tK={K}, \n\tV={V}")
    num = torch.matmul( Q, K.T ).to(dev)
    den = math.sqrt( K.shape[0] )  # sqrt of dimension of keys
    d = torch.div( num, den ).to(dev)
    if logging: print(f"d={d} = num/den = {num} / {den}")
    ret = torch.Tensor([ torch.softmax( d, 0 ) ]).to(dev)
    if logging: print(f"ret.shape={ret.shape}, V.shape={V.shape} ; ret={ret}, V={V}")
    ret = torch.matmul( ret, V ).to(dev)
    return ret

# run a toy example to test that the code at least executes w/o error.
Q,K,V = torch.Tensor([[2,3,4]]).to(device) , torch.Tensor([[4,5,6]]).to(device), torch.Tensor([[1,-1,1]]).to(device)
scaled_dot_product_attention( Q,K,V, dev=device )

# %%
def multi_head_attention( Q: torch.Tensor , K: torch.Tensor, V: torch.Tensor,
                          Wq: torch.Tensor, Wk: torch.Tensor, Wv: torch.Tensor, Wo: torch.Tensor,
                          dev: torch.device, hn: int = 8 ) -> torch.Tensor:
    """3.2.2 Multi-Head Attention
    
    i.e., map `scaled_dot_product_attention` h times into a list/array.

    head_i in the map = attention( Q.Wq_i , K.Wk_i , V.Wv_i ) for i in hn heads.

    hn = 8 by default, as seen in the paper.

    Finally, MHA multiplies the concatted heads by an output weight matrix, Wo.
    """
    if logging: print(f"[MHA] Q.shape={Q.shape}, K.shape={K.shape}, V.shape={V.shape}, Wq.shape={Wq.shape}, Wk.shape={Wk.shape}, Wv.shape={Wv.shape}, Wo.shape={Wo.shape};" +
                        f"\n\tQ={Q}, \n\tK={K}, \n\tV={V}, \n\tWq={Wq}, \n\tWk={Wk}, \n\tWv={Wv}, \n\tWo={Wo}")

    # heads = torch.concat([ 
    #     scaled_dot_product_attention( 
    #         torch.matmul(Q, Wq), 
    #         torch.matmul(K, Wk), 
    #         torch.matmul(V, Wv),
    #         dev=dev ) 
    #     for i in range(h) ])
    heads = []
    for i in range(hn):
        h = scaled_dot_product_attention( 
            torch.matmul(Q, Wq), 
            torch.matmul(K, Wk), 
            torch.matmul(V, Wv),
            dev=dev )
        heads.append(h.reshape(1))  # WARNING: unprincipled hack
    
    if logging:
        print(f"heads={heads}")
    headscc = torch.concat(heads).to(dev)

    ret = torch.matmul( headscc, Wo ).to(dev)

    return ret


hn=8
Wq, Wk, Wv = torch.randn(Q.shape[1]).to(device) , torch.randn(K.shape[1]).to(device) , torch.randn(V.shape[1]).to(device)
Wo = torch.ones( hn ).to(device)     # WARNING: paper says Wo is in the set of reals having shape h*d_v * d_model. This assignment is not that.
multi_head_attention( Q,K,V, Wq, Wk, Wv, Wo, dev=device, hn=hn )

# %%
