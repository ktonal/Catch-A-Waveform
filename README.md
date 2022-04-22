# Catch-A-Waveform

[Project Website](https://galgreshler.github.io/Catch-A-Waveform/) | [Paper](https://arxiv.org/pdf/2106.06426.pdf)

### **FORK** of the Official pytorch implementation of the paper: "Catch-A-Waveform: Learning to Generate Audio from a Single Short Example" (NeurIPS 2021)

this repo contains only training and generation. All inpainting and denoising capabilities of the original model have been removed.

## Install dependencies

```
python -m pip install -r requirements.txt
```

## Unconditional Generation

### Training

To train for unconditional inference, just place an audio signal inside the `inputs` folder
and provide its name with extension to the `input_file` property in `params.py`.
Then, from the command-line:

```
python train_main.py
```

For speech signals, set `Params.speech` to `True`:


### Inference

After training, a directory named after the input file will be created in the `outputs` folder. To inference from a
trained model simply run:

```
python generate_main.py --input_folder <model_folder_name>
```

This will generate a 30 [sec] length signal in the model folders, inside `GeneratedSignals`. To create multiple signals
with various length, you can use the `n_signals` and `length` flags, for example:

```
python generate_main.py --input_folder <model_folder_name> --n_signals 3 --length 60
```

To write signals of all scales, use the flag `--generate_all_scales`.

### Create music variations

To create variations of a given song, while enforcing general structure of the input (See sec. 4.2 in
our [paper](https://arxiv.org/pdf/2106.06426.pdf)), use the `--condition` flag:

```
python generate_main.py --input_folder <model_folder_name> --condition
```

### Run examples


Music:

```
python train_main.py --input_file TenorSaxophone_MedleyDB_185
```

Speech:

```
python train_main.py --input_file trump_farewell_address_8 --speech
```

## Pretrained Models
Instead of running the examples yourself, you can download the pretrained generators and just perform inference. After downloaing the folders, put them inside `outputs` folder and run inference.

The models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1JN2QVmuKU2rCe1nAJ7jw6DsXb4F6AKpH?usp=sharing).

## Citation

If you use this code in your research, please cite our paper:

```
@article{greshler2021catch,
  title={Catch-a-waveform: Learning to generate audio from a single short example},
  author={Greshler, Gal and Shaham, Tamar and Michaeli, Tomer},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

### Credits

The examples signals are taken from the following websites:

- Saxophone - [Medley-solos-DB](https://zenodo.org/record/1344103#.YRt7oxJRVH4).
- Speech - [VCTK Corpus](https://datashare.ed.ac.uk/handle/10283/3443).
- Trump speech - [Miller Center](https://millercenter.org/the-presidency/presidential-speeches), presidential speech
  database.
- Rock song - [FMA](https://github.com/mdeff/fma) database.
- Joseph Joachim's recording - [Josheph Joachim](https://josephjoachim.com/2013/12/11/joachim-bach-adagio-in-g-minor-1904/) website.

Some code was adapted from:

- [SinGAN](https://github.com/tamarott/SinGAN).
- Resampling - [ResizeRight](https://github.com/assafshocher/ResizeRight).
- MSS loss function - [Jukexbox](https://github.com/openai/jukebox/).
- Thanks [Federico Miotello](https://github.com/fmiotello) for the multiple holes inpainting implementation.
