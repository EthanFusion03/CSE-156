# add all  your Encoder and Decoder code here
import torch
import torch.nn as nn
import math

# Step 1: Create the enocder structure
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, max_length, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embed_size = embed_size

        # Initialize random token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        # Create 4 encoder layers
        self.layers = nn.ModuleList([
            # Define in next class
            EncoderLayer(embed_size, num_heads, dropout) 
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length
    
    def forward(self, x, attention_mask): 
        # x has size: (batch_size, seq_length)
        batch_size, seq_length = x.size()
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0).expand(batch_size, seq_length)

        # Directly add every word embedding to their position embedding
        x = self.position_embedding(positions) + self.token_embedding(x)
        x = self.dropout(x)

        attention_maps = [] ## Use to store attention maps from all layers

        # Pass through the stack of encoder layers
        for layer in self.layers:
            x, attention = layer(x, attention_mask) ##
            attention_maps.append(attention) ##

        # Now x has size: (batch_size, seq_length, embed_size)
        return x, attention_maps ##

class EncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(embed_size, num_heads, 0)
        self.feed_forward = FeedForward(embed_size, embed_size * 4, dropout)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, attention_mask):
        # Calcualte self-attention using passed in word + positional embeddings
        attn_output, attention = self.self_attn(x, attention_mask) ##
        # Connect with the residual layer and norm
        x = self.norm1(x + self.dropout1(attn_output))

        # FFN
        ff_output = self.feed_forward(x)
        # Add residual layer and norm
        x = self.norm2(x + self.dropout2(ff_output))

        return x, attention ##

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads, CLSorLM):
        super(MultiHeadSelfAttention, self).__init__()
        # Note embed_size must be divisible by num_heads since emb_size = num_heads * head_dim
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.CLSorLM = CLSorLM

        # Projection matrices for queries, keys, and values
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)

        # Final linear layer after concatenating heads
        self.fc_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, x, attention_mask=None):
        # x shape: (batch_size, seq_length, embed_size)
        batch_size, seq_length, _ = x.size()

        queries = self.query_linear(x)
        keys = self.key_linear(x)
        values = self.value_linear(x)

        # Split into multiple heads
        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim)

        # Transpose to get dimensions (batch_size, num_heads, seq_length, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Calculate attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            if self.CLSorLM == 0:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # Shape: (batch_size, 1, 1, seq_length)
            scores = scores.masked_fill(attention_mask == 0, float('-1e9'))  # Use large negative value for masking

        attention = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention, values)

        # Concatenate heads
        out = out.transpose(1,2).contiguous()  # Shape: (batch_size, seq_length, num_heads, head_dim)
        out = out.view(batch_size, seq_length, self.embed_size)

        # Final linear layer
        out = self.fc_out(out)  # Shape: (batch_size, seq_length, embed_size)

        return out, attention ##

class FeedForward(nn.Module):
    def __init__(self, embed_size, hidden_size, dropout):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embed_size)
        )
        
    def forward(self, x):
        return self.net(x)

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, max_length, num_classes, hidden_dim, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.encoder = TransformerEncoder(vocab_size, embed_size, num_heads, num_layers, max_length, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x, attention_mask):
        x, attention_maps = self.encoder(x, attention_mask) ##
        x = x.mean(dim=1) # x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits, attention_maps ##
    

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, max_length, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.embed_size = embed_size

        self.token_embbeding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(embed_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, seq_length = x.size()
        positions = torch.einsum('i,j->ij', torch.ones(batch_size, dtype=torch.long, device=x.device) , torch.arange(0, seq_length, dtype=torch.long, device=x.device))

        # Embed tokens and positions
        x = self.token_embbeding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        # Generate causal mask
        causal_mask = torch.tril(torch.ones((seq_length, seq_length), device=x.device)).unsqueeze(0).unsqueeze(0)
        causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)

        attention_maps = []

        for layer in self.layers:
            x, attention = layer(x, causal_mask)
            attention_maps.append(attention)

        return x, attention_maps

class DecoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(embed_size, num_heads, 1)
        self.feed_forward = FeedForward(embed_size, embed_size * 4, dropout)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attention_mask):
        # Masked self-attention
        attn_output, attention = self.self_attn(x, attention_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x, attention
    
class TransformerDecoderLM(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, max_length, dropout=0.1):
        super(TransformerDecoderLM, self).__init__()
        self.decoder = TransformerDecoder(vocab_size, embed_size, num_heads, num_layers, max_length, dropout)
        self.lm_head = nn.Linear(embed_size, vocab_size)
    
    def forward(self, x, targets=None):
        # x size: (batch_size, seq_length)
        decoder_output, attention_maps = self.decoder(x)
        logits = self.lm_head(decoder_output)

        if targets is not None:
            # Compute loss
            loss_fn = nn.CrossEntropyLoss()
            # Reshape logits and targets
            logits_flat = logits.view(-1, logits.size(-1)) # logits size now: (batch * seq, vocab)
            targets_flat = targets.view(-1) # (batch * seq)
            loss = loss_fn(logits_flat, targets_flat)
            return loss, attention_maps
        else:
            # Text generation
            return logits
            
        
