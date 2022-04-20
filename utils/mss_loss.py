import torch
import numpy as np


def stft(sig, n_fft, hop_length, window_size):
    s = torch.stft(sig, n_fft, hop_length, win_length=window_size,
                   window=torch.hann_window(window_size, device=sig.device), return_complex=False)
    return s


def spec(x, n_fft, hop_length, window_size):
    s = stft(x, n_fft, hop_length, window_size)
    n = torch.norm(s, p=2, dim=-1)
    return n


def norm(x):
    return (x.view(x.shape[0], -1) ** 2).sum(dim=-1).sqrt()


def squeeze(x):
    if len(x.shape) == 3:
        assert x.shape[-1] in [1, 2]
        x = torch.mean(x, -1)
    if len(x.shape) != 2:
        raise ValueError(f'Unknown input shape {x.shape}')
    return x


def multi_scale_spectrogram_loss(params, x_in, x_out):
    losses = []
    args = [params.multispec_loss_n_fft,
            params.multispec_loss_hop_length,
            params.multispec_loss_window_size]
    for n_fft, hop_length, window_size in zip(*args):
        if window_size == -1:
            window_size = x_in.shape[1]
            hop_length = window_size + 1
            n_fft = int(2 ** np.ceil(np.log2(window_size)))

        spec_in = spec(squeeze(x_in.float()), n_fft, hop_length, window_size)
        spec_out = spec(squeeze(x_out.float()), n_fft, hop_length, window_size)
        losses.append(norm(spec_in - spec_out))
    return sum(losses) / len(losses)
