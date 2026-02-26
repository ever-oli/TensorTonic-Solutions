import numpy as np

class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim

        # Xavier initialization
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (2 * hidden_dim))
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray, h_0: np.ndarray = None) -> tuple:
        """
        Forward pass through entire sequence.
        X shape: (Batch, Time, InputDim)
        Returns (y_seq, h_final).
        """
        batch_size, time_steps, _ = X.shape
        
        # 1. Initialize hidden state if not provided
        if h_0 is None:
            h_current = np.zeros((batch_size, self.hidden_dim))
        else:
            h_current = h_0

        h_list = []

        # 2. Iterate through time steps (Unrolling)
        for t in range(time_steps):
            x_t = X[:, t, :]  # (Batch, InputDim)
            
            # Compute h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
            # Weights are (Hidden, In/Hidden), so we use .T for batch multiplication
            h_current = np.tanh(x_t @ self.W_xh.T + h_current @ self.W_hh.T + self.b_h)
            h_list.append(h_current)

        # 3. Stack hidden states: (Batch, Time, HiddenDim)
        h_seq = np.stack(h_list, axis=1)
        h_final = h_current

        # 4. Output Projection: y_t = W_hy * h_t + b_y
        # Efficiently project by reshaping to (Batch * Time, HiddenDim)
        h_flat = h_seq.reshape(-1, self.hidden_dim)
        y_flat = h_flat @ self.W_hy.T + self.b_y
        
        # Reshape back to (Batch, Time, OutputDim)
        y_seq = y_flat.reshape(batch_size, time_steps, -1)

        return y_seq, h_final