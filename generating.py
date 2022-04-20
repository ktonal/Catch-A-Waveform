from utils.utils import *
from utils.plotters import *
import os


class AudioGenerator(object):
    def __init__(self, params, generators_list=None, noise_amp_list=None, reconstruction_noise_list=None):
        super(AudioGenerator, self).__init__()
        if type(params) is str:
            path = os.path.join(params, 'log.txt')
            output_folder = params
            params = params_from_log(path)
            params.output_folder = output_folder
            noise_amp_list = noise_amp_list_from_log(path)
            reconstruction_noise_list = torch.load((os.path.join(params.output_folder, 'reconstruction_noise_list.pt')),
                                                   map_location=params.device)
            generators_list = generators_list_from_folder(params)
        else:
            output_folder = params.output_folder
        self.generators_list = generators_list
        self.noise_amp_list = noise_amp_list
        self.params = params
        self.reconstruction_noise_list = reconstruction_noise_list
        self.output_folder = output_folder
        if not os.path.exists(os.path.join(output_folder, 'GeneratedSignals')):
            os.mkdir(os.path.join(output_folder, 'GeneratedSignals'))

    def generate(self, nSignals=1, length=20, generate_all_scales=False):
        for sig_idx in range(nSignals):
            # Draws a signal up to current scale, using learned generators
            output_signals_list = draw_signal(self.params, self.generators_list,
                                              [round(f * length) for f in self.params.fs_list], self.params.fs_list,
                                              self.noise_amp_list, output_all_scales=generate_all_scales)
            # Write signals
            if generate_all_scales:
                for scale_idx, sig in enumerate(output_signals_list):
                    write_signal(
                        os.path.join(self.output_folder, 'GeneratedSignals',
                                     'generated@%dHz.wav' % self.params.fs_list[scale_idx]),
                        sig, self.params.fs_list[scale_idx], overwrite=False)
            else:
                write_signal(
                    os.path.join(self.output_folder, 'GeneratedSignals',
                                 'generated@%dHz.wav' % self.params.fs_list[-1]),
                    output_signals_list, self.params.fs_list[-1], overwrite=False)

    def reconstruct(self, reconstruction_noise_list=None, write=True):
        if reconstruction_noise_list is None:
            reconstruction_noise_list = self.reconstruction_noise_list
        reconstructed_signal = draw_signal(self.params, self.generators_list,
                                           [int(l) for l in self.params.inputs_lengths],
                                           self.params.fs_list, self.noise_amp_list,
                                           reconstruction_noise_list=reconstruction_noise_list)
        if write:
            write_signal(
                os.path.join(self.output_folder, 'GeneratedSignals',
                             'reconstructed@%dHz.wav' % self.params.fs_list[-1]),
                reconstructed_signal, self.params.fs_list[-1], overwrite=False)
        else:
            return reconstructed_signal

    def extend(self, condition, filt_file=None):
        conditioned_signal = self.condition(condition, False)
        stitched_signal = time_freq_stitch_by_fft(condition['condition_signal'].squeeze().cpu().numpy(),
                                                  conditioned_signal.squeeze().cpu().numpy(),
                                                  condition['condition_fs'],
                                                  self.params.Fs, filt_file)
        output_file = os.path.join(self.output_folder, 'GeneratedSignals',
                                   condition['name'] + '_extended')
        write_signal(output_file, stitched_signal, self.params.Fs)
        return stitched_signal

    def condition(self, condition, write=True):
        condition["condition_scale_idx"] = np.where(np.array(self.params.fs_list) <= condition["condition_fs"])[0][
                                               -1] + 1
        condition["condition_signal"] = torch.Tensor(condition["condition_signal"]).expand(1, 1, -1).to(
            self.params.device)
        lengths = [int(condition["condition_signal"].shape[2] / condition["condition_fs"] * fs) for fs in
                   self.params.fs_list]
        conditioned_signal = draw_signal(self.params, self.generators_list, lengths,
                                         self.params.fs_list, self.noise_amp_list, condition=condition)
        if write:
            output_file = os.path.join(self.output_folder, 'GeneratedSignals',
                                       'conditioned_on_' + condition['name'])
            write_signal(output_file, conditioned_signal, self.params.Fs)
        else:
            return conditioned_signal
