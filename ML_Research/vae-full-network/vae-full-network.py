import numpy as np

class VAE:
    def __init__(self, input_dim: int, latent_dim: int):
        """
        Initialize VAE weight matrices and dimensions.
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = 256
        
        # Encoder Weights
        self.w_enc = np.random.randn(input_dim, self.hidden_dim) * 0.01
        self.b_enc = np.zeros(self.hidden_dim)
        
        # Latent Parameter Weights (Mean and Log-Variance)
        self.w_mu = np.random.randn(self.hidden_dim, latent_dim) * 0.01
        self.b_mu = np.zeros(latent_dim)
        self.w_log_var = np.random.randn(self.hidden_dim, latent_dim) * 0.01
        self.b_log_var = np.zeros(latent_dim)
        
        # Decoder Weights
        self.w_dec_h = np.random.randn(latent_dim, self.hidden_dim) * 0.01
        self.b_dec_h = np.zeros(self.hidden_dim)
        self.w_dec_out = np.random.randn(self.hidden_dim, input_dim) * 0.01
        self.b_dec_out = np.zeros(input_dim)
    
    def forward(self, x: np.ndarray) -> tuple:
        """
        Full forward pass: Encode -> Reparameterize -> Decode.
        Returns: (reconstruction, mu, log_var)
        """
        # 1. Encode
        h_enc = np.maximum(0, x @ self.w_enc + self.b_enc)
        mu = h_enc @ self.w_mu + self.b_mu
        log_var = h_enc @ self.w_log_var + self.b_log_var
        
        # 2. Reparameterize
        std = np.exp(0.5 * log_var)
        eps = np.random.randn(*mu.shape)
        z = mu + std * eps
        
        # 3. Decode
        h_dec = np.maximum(0, z @ self.w_dec_h + self.b_dec_h)
        logits = h_dec @ self.w_dec_out + self.b_dec_out
        x_recon = 1 / (1 + np.exp(-logits))  # Sigmoid activation
        
        return x_recon, mu, log_var
    
    def generate(self, n_samples: int) -> np.ndarray:
        """
        Generate new samples by sampling z from the prior N(0, I).
        """
        # Sample directly from the standard normal prior
        z = np.random.randn(n_samples, self.latent_dim)
        
        # Decode the sampled latent vectors
        h_dec = np.maximum(0, z @ self.w_dec_h + self.b_dec_h)
        logits = h_dec @ self.w_dec_out + self.b_dec_out
        samples = 1 / (1 + np.exp(-logits))
        
        return samples