import torch
import torch.nn as nn
from torch.nn import functional as F

## hyperparameters
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?
max_iters = 5000  # 训练步数
eval_interval = 300  ## 评估损失的训练步间隔 每300步评估一次损失
learning_rate = 1e-3
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
eval_iters = 200  ## 计算损失时的步数 即计算200次损失并求均值
n_embd = 32
# ------------
torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be trained, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))  ## 从范围内取出batch_size个数
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()  ## 类似 with torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:  ##
        losses = torch.zeros(eval_iters)  ## eval_iter 是测算200组loss
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  ## B, T, C -> B, T, head_size
        q = self.query(x)
        ## compute the attention scores
        wei = q @ k.transpose(-2, -1) * C ** -0.5  ## (B,T,head_size) @ (B, head_size, T) -> (B, T, T) 除以c**0.5
        # 是为了缩小wei的方差类似归一化
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  ##将wei的上三角不包括对角线，替换为负无穷大
        ## [:T, :T]是为了生成数据时，当输入x的token不足 block_size 时, 只取tril的T*T的小方阵来对应处理wei
        wei = F.softmax(wei, dim=-1)
        ## 对value进行加权求和
        v = self.value(x)  ## (B,T,head_size)
        out = wei @ v  ## (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_head)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)


class FeedFoward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_head, n_embd):
        super().__init__()
        head_size = n_embd // n_head
        self.sa_head = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)

    def forward(self, x):
        x = self.sa_head(x)
        x = self.ffwd(x)
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  ## 将token映射为一个n_embd维的向量，这里的token即单词索引
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  ## 将位置索引映射为一个n_emb维的向量
        self.blocks = nn.Sequential(
            Block(4, n_embd),  ## 每个block内有四个head，head输入维度 n_embd, 输出head_size 最后在将四个head输出在head_size维度相加
            Block(4, n_embd),  ## feedfoward 输入维度是n_embd, 输出也是n_embd
            Block(4, n_embd)  ## 连续三个block
        )
        self.lm_embd = nn.Linear(n_embd, vocab_size)  ##

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  ## arange生成 T个索引，返回(T, n_embd)
        x = tok_emb + pos_emb  ## 广播后则为 (B, T, n_embd)  此时x有语义信息和位置信息
        x = self.blocks(x)  ## 输出为(B,T ,n_embd)
        logits = self.lm_embd(x)  ## （B, T, vacab_size) 通过线性层映射到 vacab_size 维度得到下一个token的预测。此时的预测考虑了前文的语义信息

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # 取出idx最后block_size个token
            idx_cond = idx[:, -block_size:]  ## position_embd层输入维度是block_size， 但idx在生成过程中是time维度是逐渐增大的
            ## 不可以令其超出block_size 也就是说在生成时最多只能看到包括自身和过去在内的block_size个token
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for i in range(max_iters):
    optimizer.zero_grad()
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    loss.backward()
    optimizer.step()

    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f"step: {i} train_loss:{losses['train']:.4f}, valid_loss: {losses['val']:.4f}")

### generate data
print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=500)[0].tolist()))
