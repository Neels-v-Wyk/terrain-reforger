import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e, device=z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices


class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantizer with Exponential Moving Average (EMA) updates.
    
    This implementation helps prevent codebook collapse by:
    1. Using EMA to update embeddings instead of gradient descent
    2. Tracking usage of each code and reinitializing dead codes
    3. Using Laplace smoothing for cluster size estimation
    
    Based on VQ-VAE-2 paper and DeepMind's Sonnet implementation.
    
    Args:
        n_e: Number of embeddings (codebook size)
        e_dim: Dimension of each embedding
        beta: Commitment cost weight (only for commitment loss, not codebook)
        decay: EMA decay rate (0.99 is typical, higher = slower updates)
        eps: Epsilon for Laplace smoothing to avoid division by zero
        reset_threshold: Dead-code threshold multiplier relative to uniform usage
            (uniform usage is 1 / n_e; dead if usage_rate < reset_threshold / n_e)
        reset_interval: How often (in forward passes) to check for dead codes
    """
    
    def __init__(
        self, 
        n_e: int, 
        e_dim: int, 
        beta: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        reset_threshold: float = 0.5,
        reset_interval: int = 500
    ):
        super().__init__()
        
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.reset_threshold = reset_threshold
        self.reset_interval = reset_interval
        
        # Initialize embeddings with larger range for better initial spread
        embed = torch.randn(n_e, e_dim)
        self.register_buffer("embedding", embed)
        
        # EMA cluster sizes (N in the paper)
        self.register_buffer("cluster_size", torch.zeros(n_e))
        
        # EMA sum of encoder outputs assigned to each code (m in the paper)
        self.register_buffer("embed_avg", embed.clone())
        
        # Track forward pass count for periodic reset
        self.register_buffer("forward_count", torch.tensor(0))
        
        # Track codes that have been used at least once (for initialization)
        self.register_buffer("initialized", torch.zeros(n_e, dtype=torch.bool))

    def _uniform_usage_rate(self) -> float:
        """Expected usage per code under perfectly uniform assignment."""
        return 1.0 / float(self.n_e)

    def _dead_usage_threshold(self) -> float:
        """
        Usage-rate threshold used for dead-code decisions.

        Interprets `reset_threshold` as a multiplier of uniform usage.
        Example: reset_threshold=0.5 => threshold = 0.5 / n_e.
        """
        return float(self.reset_threshold) * self._uniform_usage_rate()
    
    def forward(self, z: torch.Tensor):
        """
        Forward pass with EMA codebook updates.
        
        Args:
            z: Encoder output (B, C, H, W)
            
        Returns:
            loss: Commitment loss only (codebook updated via EMA)
            z_q: Quantized latent (B, C, H, W)
            perplexity: Codebook usage metric
            encodings: One-hot encodings
            encoding_indices: Index of closest embedding for each position
        """
        # Reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_shape = z.shape
        z_flattened = z.view(-1, self.e_dim)  # (B*H*W, e_dim)
        
        # Compute distances: (z - e)^2 = z^2 + e^2 - 2*z*e
        d = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding ** 2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.t())
        )
        
        # Find closest embeddings
        min_encoding_indices = torch.argmin(d, dim=1)
        
        # Create one-hot encodings
        min_encodings = F.one_hot(min_encoding_indices, self.n_e).float()
        
        # Get quantized latent vectors
        z_q = F.embedding(min_encoding_indices, self.embedding).view(z_shape)
        
        # EMA updates (only during training)
        if self.training:
            self.forward_count += 1
            
            # Update cluster sizes with EMA
            encodings_sum = min_encodings.sum(0)  # Count per code
            self.cluster_size.data.mul_(self.decay).add_(
                encodings_sum, alpha=1 - self.decay
            )
            
            # Update embedding sums with EMA
            embed_sum = min_encodings.t() @ z_flattened  # Sum of z's per code
            self.embed_avg.data.mul_(self.decay).add_(
                embed_sum, alpha=1 - self.decay
            )
            
            # Laplace smoothing for stable division
            n = self.cluster_size.sum()
            cluster_size_smoothed = (
                (self.cluster_size + self.eps)
                / (n + self.n_e * self.eps)
                * n
            )
            
            # Update embeddings
            self.embedding.data.copy_(
                self.embed_avg / cluster_size_smoothed.unsqueeze(1)
            )
            
            # Mark used codes as initialized
            used_codes = encodings_sum > 0
            self.initialized.data |= used_codes
            
            # Periodic dead code reinitialization
            if self.forward_count % self.reset_interval == 0:
                self._reinitialize_dead_codes(z_flattened)
        
        # Commitment loss (encoder learns to commit to codebook)
        # No codebook loss since we use EMA updates
        commitment_loss = self.beta * F.mse_loss(z_q.detach(), z)
        
        # Straight-through estimator: copy gradients from z_q to z
        z_q = z + (z_q - z).detach()
        
        # Compute perplexity (effective codebook usage)
        avg_probs = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Reshape back to original spatial dimensions
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        return (
            commitment_loss, 
            z_q, 
            perplexity, 
            min_encodings, 
            min_encoding_indices.unsqueeze(1)
        )
    
    def _reinitialize_dead_codes(self, z_flattened: torch.Tensor):
        """
        Reinitialize codes that haven't been used enough.
        
        Dead codes are reinitialized to random encoder outputs from the
        current batch, giving them a chance to be useful.
        """
        # Find dead codes (usage below threshold)
        usage_rate = self.cluster_size / (self.cluster_size.sum() + 1e-10)
        dead_threshold = self._dead_usage_threshold()
        dead_codes = usage_rate < dead_threshold
        
        # Also consider codes that were never initialized
        dead_codes = dead_codes | ~self.initialized
        
        n_dead = dead_codes.sum().item()
        
        if n_dead > 0 and z_flattened.shape[0] > 0:
            # Sample random encoder outputs to reinitialize dead codes
            n_samples = min(n_dead, z_flattened.shape[0])
            
            # Random indices from current batch
            rand_indices = torch.randperm(z_flattened.shape[0])[:n_samples]
            rand_samples = z_flattened[rand_indices]
            
            # Get indices of dead codes
            dead_indices = torch.where(dead_codes)[0][:n_samples]
            
            # Reinitialize embeddings
            self.embedding.data[dead_indices] = rand_samples
            self.embed_avg.data[dead_indices] = rand_samples
            self.cluster_size.data[dead_indices] = 1.0  # Small initial count
            self.initialized.data[dead_indices] = True
    
    def get_codebook_usage_stats(self) -> dict:
        """Return statistics about codebook usage for monitoring."""
        usage_rate = self.cluster_size / (self.cluster_size.sum() + 1e-10)
        uniform_rate = self._uniform_usage_rate()
        dead_threshold = self._dead_usage_threshold()

        # Shannon entropy of usage distribution, normalized to [0, 1]
        entropy = -torch.sum(usage_rate * torch.log(usage_rate + 1e-10))
        max_entropy = np.log(self.n_e)
        entropy_normalized = float((entropy / max_entropy).item()) if max_entropy > 0 else 0.0

        # Concentration stats (share captured by top-k codes)
        sorted_usage, _ = torch.sort(usage_rate, descending=True)
        top1_share = sorted_usage[:1].sum().item()
        top5_share = sorted_usage[:5].sum().item()
        top10_share = sorted_usage[:10].sum().item()
        
        return {
            "uniform_rate": uniform_rate,
            "dead_threshold": dead_threshold,
            "active_codes": (usage_rate > dead_threshold).sum().item(),
            "dead_codes": (usage_rate <= dead_threshold).sum().item(),
            "active_codes_above_uniform": (usage_rate > uniform_rate).sum().item(),
            "active_codes_above_half_uniform": (usage_rate > (0.5 * uniform_rate)).sum().item(),
            "entropy": entropy.item(),
            "entropy_normalized": entropy_normalized,
            "top1_share": top1_share,
            "top5_share": top5_share,
            "top10_share": top10_share,
            "max_usage": usage_rate.max().item(),
            "min_usage": usage_rate.min().item(),
            "usage_std": usage_rate.std().item(),
        }
