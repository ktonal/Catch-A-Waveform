from utils.utils import *
import glob
from params import Params
from training import train
from utils.plotters import *
import time
from datetime import datetime
from generating import AudioGenerator


startTime = time.time()
params = Params()


if params.is_cuda:
    torch.cuda.set_device(params.gpu_num)
    params.device = torch.device("cuda:%d" % params.gpu_num)

# Get input signal
samples = get_input_signal(params)
# set scales
params.fs_list = [f for f in params.fs_list if f <= params.Fs]
if params.fs_list[-1] != params.Fs:
    params.fs_list.append(params.Fs)
params.scales = [params.Fs / f for f in params.fs_list]

print('Working on file: %s' % params.input_file)

# Set params by run_node and signal type
params.scheduler_milestones = [int(params.num_epochs * 2 / 3)]
if params.speech:
    params.alpha1 = 10
    params.alpha2 = 0
    params.add_cond_noise = False
else:
    params.alpha1 = 0
    params.alpha2 = 1e-4
    params.add_cond_noise = True

params.dilation_factors = [2 ** i for i in range(params.num_layers)]

# Create output folder
if not os.path.exists('outputs'):
    os.mkdir('outputs')

if os.path.exists(params.output_folder):
    dirs = glob.glob(params.output_folder + '*')
    params.output_folder = params.output_folder + '_' + str(len(dirs) + 1)
os.mkdir(params.output_folder)
print('Writing results to %s\n' % params.output_folder)

# samples = samples.reshape((1, -1))

# Create input signal for each scale
signals_list, fs_list = create_input_signals(params, torch.tensor(samples), params.Fs)
if len(signals_list) == 0:
    # Fall back if no first scale was found
    params.set_first_scale_by_energy = False
    params.scales = params.scales[2:]  # Manually start from 500
    signals_list, fs_list = create_input_signals(params, torch.tensor(samples), params.Fs)
params.scales = [params.Fs / f for f in fs_list]
params.fs_list = fs_list
params.inputs_lengths = [len(s) for s in signals_list]

# Write parameters of run to a text file
with open(os.path.join(params.output_folder, 'log.txt'), 'w') as f:
    f.write(''.join(["%s = %s\n" % (k, v) for k, v in params.__dict__.items()]))

print('Running on ' + str(params.device))

# Start training
output_signals, loss_vectors, generators_list, noise_amp_list, energy_list, reconstruction_noise_list = train(
    params, signals_list)

# Save reconstruction noise list
torch.save(reconstruction_noise_list, os.path.join(params.output_folder, 'reconstruction_noise_list.pt'))

with open(os.path.join(params.output_folder, 'log.txt'), 'a') as f:
    f.write('\nTotal Runtime is: %d minutes' % ((time.time() - startTime) / 60))
    f.write('\n Finished running in : %s' % datetime.fromtimestamp(time.time()))

##############
# Generating #
##############
audio_generator = AudioGenerator(params, generators_list, noise_amp_list,
                                 reconstruction_noise_list=reconstruction_noise_list)
audio_generator.generate()
audio_generator.reconstruct()

#################
# Plotting Area #
#################
# Plot Signals
if params.plot_signals:
    os.mkdir(os.path.join(params.output_folder, 'figures'))
    for real_signal, outputs, fs in zip(signals_list, output_signals, params.fs_list):
        output_file(os.path.join(params.output_folder, 'figures', '%dHz' % fs))
        plot_signal_time_freq(real_signal, outputs['reconstructed_signal'], outputs['fake_signal'], Fs=fs,
                              labels=['Real Signal', 'Reconstructed Signal', 'Fake Signal'])
# Plot losses
if params.plot_losses:
    if not os.path.exists(os.path.join(params.output_folder, 'figures')):
        os.mkdir(os.path.join(params.output_folder, 'figures'))
    plot_losses(params, loss_vectors)
