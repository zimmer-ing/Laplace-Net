# MIT License, Copyright (c) 2025 Bernd Zimmering
# See LICENSE file for details.
# Please cite: Zimmering, B. et al. (2025), "Breaking Free: Decoupling Forced Systems with Laplace Neural Networks", ECML PKDD, Porto.


import torch


def discrete_laplace_transform_1D(x, s_in, t):
    """
    Computes the Laplace transform of a signal x(t) for multiple values of s.

    Args:
        x (Tensor): Shape (Batch, Num_t, 1) - Signal
        s_in (Tensor): Shape (Batch, Num_s, Num_features,Num_Terms) - Laplace variable
        t (Tensor): Shape (Batch, Num_t,1) - Time vector

    Returns:
        Tensor of shape (Batch, Num_s, Num_features)
    """
    dt = t[0,1,0]-t[0,0,0]
    batch_dim = x.shape[0]
    s_res = torch.zeros(*s_in.shape, device=x.device, dtype=s_in.dtype)

    for sample in range(batch_dim):
        s_s = s_in[sample].view(-1, 1)  # (Num_s, 1)
        t_s = t[sample].view(1, -1)  # (1, Num_t)
        kernel = torch.exp(-s_s * t_s)  # (Num_s, Num_t)
        x_s = x[sample].unsqueeze(0).squeeze(-1)  # (1, Num_t)
        result = torch.sum(x_s * kernel, dim=-1) * dt
        s_res[sample] = result.view(1,s_in.shape[1],1, s_in.shape[3])
    return s_res

def discrete_laplace_transform(x, s_in, t):
    """
    Computes the Laplace transform of a signal x(t) for multiple values of s.

    Args:
        x (Tensor): Shape (Batch, Num_t, Features) - Signal (real-valued)
        s_in (Tensor): Shape (Batch, Num_s, 1, Num_Terms) - Laplace variable (complex).
                       Same s-values for all output features.
        t (Tensor): Shape (Batch, Num_t, 1) - Time vector (real)

    Returns:
        Tensor of shape (Batch, Num_s, Num_features, Num_Terms)
    """
    dt = t[0, 1, 0] - t[0, 0, 0]
    B, N, F = x.shape
    _, S, _, T_terms = s_in.shape

    s_res = torch.zeros(B, S, F, T_terms, device=x.device, dtype=s_in.dtype)

    for b in range(B):
        t_s = t[b].view(1, -1)  # (1, N)
        x_b = x[b]  # (N, F)

        # Flatten s for current batch: (S * T_terms, 1)
        s_flat = s_in[b].view(-1, 1)

        # Kernel: (S * T_terms, N)
        kernel = torch.exp(-s_flat * t_s)

        # Multiply with all features at once
        # Expand x_b -> (1, N, F)
        x_exp = x_b.unsqueeze(0)  # (1, N, F)
        kernel_exp = kernel.unsqueeze(-1)  # (S*T_terms, N, 1)

        # Element-wise multiplication and integration over time
        result = torch.sum(x_exp * kernel_exp, dim=1) * dt  # (S*T_terms, F)

        # Reshape back: (S, 1, T_terms, F) -> (S, F, T_terms)
        result = result.view(S, T_terms, F).permute(0, 2, 1)

        s_res[b] = result

    return s_res.permute(0, 1, 3, 2)


def laplace_from_fft_1D(x_t, s, t, chunk_size=1000, mode="sample"):
    """
    Computes the Laplace transform X(s) via Fourier coefficients and accounts for a start time shift t_0.

    Supported modes:
      - `mode="chunk"`: Memory-efficient processing with chunks (if `chunk_size > 0`).
      - `mode="batch"`: Vectorized for the entire batch (memory-intensive but faster).
      - `mode="sample"`: Uses a `for` loop over `B` (sample-wise).

    Assumed shapes:
      x_t: (B, N, 1)  -> Time domain signal (real-valued)
      s:   (B, num_t, feature, ilt_terms) -> Query points in the Laplace domain (complex)
      t:   (B, N, 1)  -> Time vector (irregular start time per batch allowed)

    Args:
        chunk_size (int): Number of `S` values per processing step (`mode="chunk"`).
        mode (str): `"chunk"`, `"batch"`, or `"sample"`.

    Returns:
        Xs: (B, num_t, feature, ilt_terms)
    """
    # ------------------------------------------------
    # 1) Preparations & Reshaping of Laplace variables
    # ------------------------------------------------
    B, num_t_, feature_, ilt_terms_ = s.shape
    S = num_t_ * feature_ * ilt_terms_  # Total number of Laplace query points

    # Time difference dt
    dt = t[0, 1, 0] - t[0, 0, 0]

    # Extract start time t0 per batch
    t0 = t[:, 0, 0].view(B, 1)  # Shape (B, 1) for easy broadcasting

    # x_t -> (B, N)
    x_t_2d = x_t.squeeze(-1)  # (B, N)

    # ------------------------------------------------
    # 2) FFT + Define frequencies
    # ------------------------------------------------
    N = x_t_2d.shape[-1]

    # Compute the FFT and scale correctly with dt
    alpha = torch.fft.fft(x_t_2d, dim=-1) / (N * dt)  # => (B, N)

    # Frequency values for the FFT
    w_fft = -2.0 * torch.pi * torch.fft.fftfreq(N, d=dt).to(alpha.device)  # (N,)

    # ------------------------------------------------
    # 3) Computation of X(s) - Three modes: Chunk, Batch, Sample
    # ------------------------------------------------

    # Reshape `s` for later matrix multiplication
    s_reshaped = s.view(B, S)  # (B, S)

    if mode == "chunk" and S > chunk_size:
        # -------------- CHUNK-WISE PROCESSING --------------
        S_chunks = torch.chunk(s_reshaped, chunks=max(1, S // chunk_size), dim=1)  # List of `(B, S_chunk)`
        Xs_chunks = []

        for s_chunk in S_chunks:
            # Compute `denom` (B, S_chunk, N)
            denom_chunk = s_chunk.unsqueeze(-1) + 1j * w_fft.view(1, 1, N)

            # Matrix multiplication instead of explicit summation
            Xs_chunk = torch.matmul(1 / denom_chunk, alpha.unsqueeze(-1)).squeeze(-1) * dt  # (B, S_chunk)

            # Correction for start time t_0 with exp(-s t_0)
            shift_factor = torch.exp(-s_chunk * t0)  # (B, S_chunk)
            Xs_chunk *= shift_factor

            Xs_chunks.append(Xs_chunk)

        # Concatenate back
        Xs = torch.cat(Xs_chunks, dim=1)  # (B, S)

    elif mode == "batch":
        # -------------- BATCH-WISE PROCESSING (Fully vectorized) --------------
        # Compute `denom` (B, S, N)
        denom = s_reshaped.unsqueeze(-1) + 1j * w_fft.view(1, 1, N)

        # Matrix multiplication instead of explicit summation
        Xs = torch.matmul(1 / denom, alpha.unsqueeze(-1)).squeeze(-1) * dt  # (B, S)

        # Correction for start time t_0 with exp(-s t_0)
        shift_factor = torch.exp(-s_reshaped * t0)  # (B, S)
        Xs *= shift_factor

    elif mode == "sample":
        # -------------- SAMPLE-WISE PROCESSING (With `for` loop over `B`) --------------
        Xs_list = []
        for b_i in range(B):
            alpha_b = alpha[b_i]  # shape (N,)
            s_b = s_reshaped[b_i].unsqueeze(-1)  # (S,1)
            w_b = w_fft.unsqueeze(0)  # (1,N)

            # Complex denominator for each frequency
            denom = s_b + 1j * w_b  # (S,N)

            # Sum over all frequencies
            X_b = (alpha_b.unsqueeze(0) / denom).sum(dim=-1) * dt  # shape(S,)

            # Correction for start time t_0 with exp(-s t_0)
            shift_factor = torch.exp(-s_b * t0[b_i])  # (S,1)
            X_b *= shift_factor.squeeze(-1)  # Keep shape (S,)

            Xs_list.append(X_b)

        # Stack => (B, S)
        Xs = torch.stack(Xs_list, dim=0)

    else:
        raise ValueError("Invalid mode. Use 'chunk', 'batch', or 'sample'.")

    # Reshape back to (B, num_t_, feature_, ilt_terms_)
    Xs = Xs.view(B, num_t_, feature_, ilt_terms_)

    return Xs


def laplace_from_fft(x_t, s, t, chunk_size=1000, mode="sample"):
    """
    Computes the Laplace transform X(s) via Fourier coefficients and applies a start time shift t0.
    This version supports multidimensional time signals (multiple features).

    Supported modes:
      - mode="chunk": Memory-efficient processing in chunks (if total S > chunk_size).
      - mode="batch": Fully vectorized for the entire batch (faster but more memory-intensive).
      - mode="sample": Loop over batch samples.

    Assumed shapes:
      x_t: (B, N, F)  -> Time domain signal (real-valued) with F features.
      s:   (B, num_s, 1, ilt_terms) -> Laplace query points (complex)
           (identical s-values for all features).
      t:   (B, N, 1) -> Time vector (real)

    Returns:
      Xs: (B, num_s, ilt_terms, F)
    """
    import torch

    B, num_s, one, ilt_terms = s.shape  # s: (B, num_s, 1, ilt_terms)
    F = x_t.shape[-1]  # Number of features in x_t
    dt = t[0, 1, 0] - t[0, 0, 0]  # Time step (assumed constant)

    # Extract the start time t0 for each batch (for the shift)
    t0 = t[:, 0, 0].view(B, 1)  # (B, 1)

    # Number of time points
    N = x_t.shape[1]

    # Perform FFT over the time axis (dim=1) for each feature in x_t (shape: (B, N, F)).
    # The FFT result is then scaled by (N * dt).
    alpha = torch.fft.fft(x_t, dim=1) / (N * dt)  # Result: (B, N, F), complex

    # FFT frequencies (w_fft):
    w_fft = -2.0 * torch.pi * torch.fft.fftfreq(N, d=dt).to(alpha.device)  # shape: (N,)

    # Treat s as a set of query points per batch.
    # Originally, s has shape (B, num_s, 1, ilt_terms), i.e. total S = num_s * ilt_terms query points.
    S = num_s * ilt_terms
    s_reshaped = s.view(B, S)  # (B, S)

    # --- Compute X(s) for the different modes ---
    if mode == "chunk" and S > chunk_size:
        # CHUNK-WISE PROCESSING:
        Xs_chunks = []
        # Split s_reshaped into chunks along the S dimension.
        S_chunks = torch.chunk(s_reshaped, chunks=max(1, S // chunk_size), dim=1)  # List of (B, S_chunk)
        for s_chunk in S_chunks:
            # Create the denominator:
            # s_chunk: (B, S_chunk) -> unsqueeze: (B, S_chunk, 1)
            # w_fft: (N,) -> reshape to (1, 1, N)
            # Resulting denominator: (B, S_chunk, N)
            denom_chunk = s_chunk.unsqueeze(-1) + 1j * w_fft.view(1, 1, N)

            # alpha: (B, N, F)
            # Matrix multiplication: (B, S_chunk, N) x (B, N, F) -> (B, S_chunk, F)
            Xs_chunk = torch.matmul(1 / denom_chunk, alpha) * dt  # (B, S_chunk, F)

            # Apply the start time shift factor: exp(-s * t0)
            # s_chunk: (B, S_chunk); t0: (B, 1) → shift_factor: (B, S_chunk)
            shift_factor = torch.exp(-s_chunk * t0)
            Xs_chunk = Xs_chunk * shift_factor.unsqueeze(-1)  # Broadcast over F

            Xs_chunks.append(Xs_chunk)
        # Concatenate the chunks: (B, S, F)
        Xs = torch.cat(Xs_chunks, dim=1)

    elif mode == "batch":
        # FULLY VECTORIZED BATCH MODE:
        # Create the denominator for all s-values:
        # s_reshaped.unsqueeze(-1): (B, S, 1)  +  1j * w_fft.view(1, 1, N) gives (B, S, N)
        denom = s_reshaped.unsqueeze(-1) + 1j * w_fft.view(1, 1, N)
        # Matrix multiplication: (B, S, N) x (B, N, F) -> (B, S, F)
        Xs = torch.matmul(1 / denom, alpha) * dt
        # Apply the shift factor:
        shift_factor = torch.exp(-s_reshaped * t0)  # (B, S)
        Xs = Xs * shift_factor.unsqueeze(-1)  # (B, S, F)

    elif mode == "sample":
        # SAMPLE-WISE PROCESSING (process each batch sample individually):
        Xs_list = []
        for b in range(B):
            # alpha_b: (N, F)
            alpha_b = alpha[b]
            # s_b: (S,) -> unsqueeze to (S, 1)
            s_b = s_reshaped[b].unsqueeze(-1)
            # w_fft: (N,) -> unsqueeze to (1, N)
            w_b = w_fft.unsqueeze(0)
            # Denom: (S, N)
            denom = s_b + 1j * w_b
            # Element-wise division:
            # alpha_b: (N, F) → unsqueeze to (1, N, F)
            # denom: (S, N) → unsqueeze to (S, N, 1)
            # Sum over time (dim=1) to get (S, F)
            X_b = (alpha_b.unsqueeze(0) / denom.unsqueeze(-1)).sum(dim=1) * dt
            # Apply the shift factor:
            shift_factor = torch.exp(-s_b * t0[b])  # (S, 1)
            X_b = X_b * shift_factor  # (S, F)
            Xs_list.append(X_b)
        # Stack along the batch dimension: (B, S, F)
        Xs = torch.stack(Xs_list, dim=0)

    else:
        raise ValueError("Invalid mode. Use 'chunk', 'batch', or 'sample'.")

    # --- Reshape the result ---
    # Currently, Xs has shape (B, S, F) with S = num_s * ilt_terms.
    # We reshape Xs into (B, num_s, ilt_terms, F) so that the feature dimension F becomes the last dimension.
    Xs = Xs.view(B, num_s, ilt_terms, F)
    return Xs
