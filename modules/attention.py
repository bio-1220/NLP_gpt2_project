import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):

    ### YOUR CODE HERE
    # 1. Calculate the attention score
    attn_score = torch.matmul(query, key.transpose(-2, -1))
    attn_score = attn_score / (self.attention_head_size ** 0.5)

    # 2. Causal mask: make sure each token only attends to previous tokens
    seq_len = query.size(-2)
    i = torch.arange(seq_len, device=query.device).view(seq_len, 1)
    j = torch.arange(seq_len, device=query.device).view(1, seq_len)
    causal_mask = j > i
    attn_score = attn_score.masked_fill(causal_mask, float("-inf"))

    # 3. Padding Mask: make sure padding tokens do not contribute to attention score
    if attention_mask is not None:
      attn_score = attn_score + attention_mask

    # 4. Softmax
    attn_score = torch.softmax(attn_score, dim=-1)

    # 5. Dropout
    attn_score = self.dropout(attn_score)

    # 6. Value multiplication
    attn_value = torch.matmul(attn_score, value)

    # 7. Merge head
    attn_value = rearrange(attn_value, 'b h t d -> b t (h d)')

    return attn_value


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
