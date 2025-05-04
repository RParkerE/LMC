import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import numpy as np

class LMC(nn.Module):
    def __init__(self, vocab_size, metadata_size, embedding_dim=100, hidden_dim=64, dropout=0.2):
        super(LMC, self).__init__()
        
        # Model network parameters
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.metadata_embedding = nn.Embedding(metadata_size, embedding_dim)
        
        # Model network (p_theta)
        self.model_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim * 2)  # Outputs mu and log_sigma
        )
        
        # Variational network (q_phi)
        self.context_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
            batch_first=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.variational_net = nn.Sequential(
            nn.Linear(embedding_dim * 2 + hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim * 2)  # Outputs mu and log_sigma
        )
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, vocab_size)
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
    def encode_context(self, context_words):
        # context_words: [batch_size, seq_len]
        context_emb = self.word_embedding(context_words)  # [batch_size, seq_len, embedding_dim]
        
        # BiLSTM encoding
        lstm_out, _ = self.context_encoder(context_emb)  # [batch_size, seq_len, hidden_dim*2]
        
        # Attention
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)  # [batch_size, seq_len, 1]
        context_encoding = torch.sum(attention_weights * lstm_out, dim=1)  # [batch_size, hidden_dim*2]
        
        return context_encoding
    
    def model_network(self, word_idx, metadata_idx):
        # Get embeddings
        word_emb = self.word_embedding(word_idx)  # [batch_size, embedding_dim]
        metadata_emb = self.metadata_embedding(metadata_idx)  # [batch_size, embedding_dim]
        
        # Concatenate
        combined = torch.cat([word_emb, metadata_emb], dim=1)  # [batch_size, embedding_dim*2]
        
        # Get distribution parameters
        params = self.model_net(combined)  # [batch_size, embedding_dim*2]
        mu_p, log_sigma_p = torch.chunk(params, 2, dim=1)
        sigma_p = torch.exp(log_sigma_p)
        
        return mu_p, sigma_p
    
    def variational_network(self, word_idx, metadata_idx, context_words):
        # Get word and metadata embeddings
        word_emb = self.word_embedding(word_idx)  # [batch_size, embedding_dim]
        metadata_emb = self.metadata_embedding(metadata_idx)  # [batch_size, embedding_dim]
        
        # Encode context
        context_encoding = self.encode_context(context_words)  # [batch_size, hidden_dim*2]
        
        # Combine all inputs
        combined = torch.cat([word_emb, metadata_emb, context_encoding], dim=1)
        
        # Get distribution parameters
        params = self.variational_net(combined)
        mu_q, log_sigma_q = torch.chunk(params, 2, dim=1)
        sigma_q = torch.exp(log_sigma_q)
        
        return mu_q, sigma_q
    
    def forward(self, word_idx, metadata_idx, context_words, target_words):
        # Get distributions
        mu_p, sigma_p = self.model_network(word_idx, metadata_idx)
        mu_q, sigma_q = self.variational_network(word_idx, metadata_idx, context_words)
        
        # Sample from variational posterior
        eps = torch.randn_like(mu_q)
        z = mu_q + sigma_q * eps
        
        # Compute KL divergence
        p_dist = Normal(mu_p, sigma_p)
        q_dist = Normal(mu_q, sigma_q)
        kl_div = kl_divergence(q_dist, p_dist).mean()
        
        # Reconstruct context words
        logits = self.output_proj(z)  # [batch_size, vocab_size]
        reconstruction_loss = F.cross_entropy(logits, target_words)
        
        # Total loss
        loss = reconstruction_loss + kl_div
        
        return {
            'loss': loss,
            'reconstruction_loss': reconstruction_loss,
            'kl_div': kl_div,
            'z': z
        }
    
    def expand_acronym(self, acronym_idx, metadata_idx, context_words, candidate_expansions):
        # Get variational posterior
        mu_q, sigma_q = self.variational_network(acronym_idx, metadata_idx, context_words)
        
        # Get model distribution for each candidate
        expansion_scores = []
        for expansion_idx in candidate_expansions:
            mu_p, sigma_p = self.model_network(expansion_idx, metadata_idx)
            
            # Compute KL divergence
            p_dist = Normal(mu_p, sigma_p)
            q_dist = Normal(mu_q, sigma_q)
            kl_div = kl_divergence(q_dist, p_dist).mean()
            
            expansion_scores.append(kl_div)
        
        # Return best expansion (lowest KL divergence)
        best_idx = torch.argmin(torch.stack(expansion_scores))
        return candidate_expansions[best_idx] 