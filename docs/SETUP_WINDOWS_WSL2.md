# Windows + RTX 3060 Ti setup (WSL2)

This guide reproduces the upstream AudioDec 48 kHz VCTK v1 file codec demo first. It does not yet install or train the PopCornConvVoice VC model.

## Model selection

Start with the matching official pair:

- Encoder: `exp/autoencoder/symAD_vctk_48000_hop300/checkpoint-200000steps.pkl`
- Vocoder: `exp/vocoder/AudioDec_v1_symAD_vctk_48000_hop300_clean/checkpoint-500000steps.pkl`
- AudioDec demo model name: `vctk_v1`

For the later speed comparison, use `vctk_v2` with the same encoder and the v2 vocoder checkpoint. Do not mix v1/v2 checkpoint families.

Official checkpoint archive: https://github.com/facebookresearch/AudioDec/releases/download/pretrain_models_v02/exp.zip

## 1. Check Windows host

In Windows PowerShell:

```powershell
nvidia-smi
wsl --status
wsl -l -v
py -0p
```

The NVIDIA driver must support CUDA in WSL2. Do not install a separate Linux NVIDIA kernel driver inside WSL. The CUDA runtime bundled with the PyTorch wheel is sufficient for this inference test.

## 2. Install / open Ubuntu 22.04 in WSL2

If WSL is not installed, from an elevated PowerShell:

```powershell
wsl --install -d Ubuntu-22.04
```

Restart if Windows requests it, launch Ubuntu, and finish its first-run username setup. Confirm the distro uses WSL2:

```powershell
wsl -l -v
```

## 3. Install the upstream project and checkpoints

In the Ubuntu WSL terminal:

```bash
sudo apt update
sudo apt install -y git wget unzip python3.10 python3.10-venv python3-pip libsndfile1 ffmpeg

mkdir -p ~/work
cd ~/work
git clone https://github.com/facebookresearch/AudioDec.git
cd AudioDec
git checkout pretrain_models_v02

wget -O exp.zip https://github.com/facebookresearch/AudioDec/releases/download/pretrain_models_v02/exp.zip
unzip -q exp.zip
```

The archive should create the `exp/` directory inside the AudioDec checkout.

## 4. Create a pinned Python environment

In the same Ubuntu terminal, from `~/work/AudioDec`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# CUDA 12.1 PyTorch wheels; compatible with RTX 30-series GPUs and a sufficiently recent Windows NVIDIA driver.
python -m pip install torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Install AudioDec runtime dependencies without allowing requirements.txt to replace the pinned torch/torchaudio.
python -m pip install numpy==1.26.4 soundfile sounddevice scikit-learn librosa==0.10.2.post1 pyyaml tqdm tensorboardX
```

The upstream repository documents Python 3.8+, CUDA 11.0+, and PyTorch 1.10+ and was tested on Ubuntu 20.04/V100. The versions above are a reproducible starting point for this RTX 3060 Ti setup, not an upstream-certified Windows/WSL matrix. If an upstream dependency fails, record the complete error before changing versions.

## 5. Verify CUDA

```bash
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND')"
```

Expected: `CUDA available: True` and the RTX 3060 Ti name. If false, stop here and resolve WSL2 GPU passthrough / Windows driver before installing more packages.

## 6. Run the 48 kHz AudioDec v1 reconstruction test

Use a mono 48 kHz WAV file that you own or have permission to process. If needed, convert a file:

```bash
ffmpeg -i input.wav -ar 48000 -ac 1 input_48k_mono.wav
```

Then:

```bash
python demoFile.py --model vctk_v1 -i input_48k_mono.wav -o reconstructed_v1.wav
```

Compare the input and reconstructed output. This is a codec reconstruction test, not voice conversion.

For the v2 comparison:

```bash
python demoFile.py --model vctk_v2 -i input_48k_mono.wav -o reconstructed_v2.wav
```

## 7. Verify checkpoint paths

```bash
ls -lh exp/autoencoder/symAD_vctk_48000_hop300/checkpoint-200000steps.pkl
ls -lh exp/vocoder/AudioDec_v1_symAD_vctk_48000_hop300_clean/checkpoint-500000steps.pkl
ls -lh exp/vocoder/AudioDec_v2_symAD_vctk_48000_hop300_clean/checkpoint-500000steps.pkl
```

## What to send back

Send the output of:

1. Windows `nvidia-smi`
2. Windows `wsl -l -v`
3. WSL PyTorch CUDA verification command
4. The complete output or traceback from `demoFile.py`
5. Whether `reconstructed_v1.wav` was created and plays correctly

Do not start full training yet. First establish the official pretrained reconstruction baseline and measure it on the target machine.
