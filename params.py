import torch
import dataclasses as dtc


@dtc.dataclass
class Params:

    input_file = "TenorSaxophone_MedleyDB_185.wav"

    ######################
    # Running Parameters #
    ######################
    start_time: int = 0
    min_length: int = 20
    max_length: int = 60
    plot_signals: bool = False
    plot_losses: bool = False
    init_sample_rate: int = 16000
    fs_list = [
        # 320,
        # 400,
        500,
        # 640,
        # 800,
        1000,
        # 1280,
        # 1600,
        2000,
        # 2500,
        4000,
        8000,
        # 10000,
        # 12000,
        # 14400,
        16000
    ]
    run_mode = 'normal'
    speech : bool = False
    set_first_scale_by_energy: bool = True
    add_cond_noise: bool = True
    min_energy_th: float = 0.0025  # minimum mean energy for first scale
    is_cuda: bool = torch.cuda.is_available()
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    initial_noise_amp: float = 1
    noise_amp_factor: float = 0.005

    #####################
    # Losses Parameters #
    #####################
    lambda_grad: float = 0.01
    alpha1: float = 10
    alpha2: float = 1e-4
    multispec_loss_n_fft = (2048, 1024, 512)
    multispec_loss_hop_length = (240, 120, 50)
    multispec_loss_window_size = (1200, 600, 240)

    ###########################
    # Optimization Parameters #
    ###########################
    num_epochs: int = 2000
    learning_rate: float = 0.0005
    scheduler_lr_decay: float = 0.05
    beta1: float = 0.5

    ####################
    # Model Parameters #
    ####################
    filter_size: int = 9  # was 9
    num_layers: int = 6  # was 8
    hidden_channels_init: int = 16   # was 16
    growing_hidden_channels_factor: int = 4  # was 6

    #######################
    # Set during training #
    #######################
    gpu_num = 0
    # target sampling rate
    Fs = 16000
    # downsampling factors
    scales = []
    # ResizeLayer for resampling
    resamplers = {}
    # dim dilation
    dilation_factors = []