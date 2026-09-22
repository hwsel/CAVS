Computation-Adaptive Video Streaming (CAVS) - MMSP 2026

The project, computation-adaptive video streaming mechanism, namely CAVS, transitions bandwidth-adaptive streaming to computation-adaptive by leveraging the recent advancements in generative AI technologies (e.g., super-resolution). CAVS stores and streams only the low resolution video and adopts a computation-adaptive, saliency based partial super-resolution approach on the client to maximize the perceptual video quality. Furthermore, we employ a Proportional-Derivative (PD) controller to adapt the super resolution workload per frame, ensuring real-time playback under varying computation capacities. For an in-depth explanation of the proposed system, please refer to the MMSP '26 paper:

Zichen Zhu, Stefano Petrangeli, Yao Liu, and Sheng Wei. Computation-Adaptive Video Streaming. IEEE International Workshop on Multimedia Signal Processing (MMSP), September 2026.

# Computation-Adaptive Video Streaming (CAVS)

Source code for CAVS published at MMSP 2026.

## 1. External Repositories & Datasets

* **ESPCN:** [leftthomas/ESPCN](https://github.com/leftthomas/ESPCN) — Efficient Sub-Pixel Convolutional Neural Network for real-time video super-resolution.
* **EVASR:** [symmru/EVASR_MMsys2023](https://github.com/symmru/EVASR_MMsys2023) — Generates 6x6 tile-based saliency weight matrices (`.npy`).
* **STVS:** [guotaowang/STVS](https://github.com/guotaowang/STVS) — Spatio-temporal video saliency detection model.
* **UVG Dataset:** [Ultra Video Group (UVG)](https://ultravideo.fi/dataset.html) — 4K sequences (e.g., `Beauty_3840x2160.yuv`).

## 2. System Requirements

* **OS:** Linux (Ubuntu 20.04 / 22.04 recommended)
* **GPU:** NVIDIA GPU with CUDA support (8GB+ VRAM recommended for 4K processing)
* **NVIDIA Driver & CUDA:** CUDA 12.8 compatible driver
* **FFmpeg Build:** Must be compiled with NVENC and CUDA support (`hevc_nvenc`, `libvmaf_cuda`)
* **RAM:** 16GB+ recommended (for handling raw uncompressed YUV 4K frames)

## 3. Environment Setup

### Step 1: Create & Activate Environment

```bash
conda create -n cavs python=3.10 -y
conda activate cavs
```

### Step 2: Docker Image (For VMAF CUDA Evaluation)

Build the Docker image once to provide the CUDA-accelerated FFmpeg environment required by `live_demo.sh`:

```bash
docker build -t vmaf-cuda:latest .
```

### Step 3: Install PyTorch & Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

```bash
pip install opencv-python imageio pillow numpy pandas scipy matplotlib flask requests tabulate
```

**System Requirement:** FFmpeg built with CUDA (`libvmaf_cuda`) support must be installed in the system `PATH`.

## 4. Directory Structure

Download the 4K raw YUV test videos from the [UVG Dataset Website](https://ultravideo.fi/dataset.html) and ensure your directory is organized as follows:

```text
├── live_demo.py
├── live_demo.sh
├── Dockerfile                    # Docker build for vmaf_cuda
├── WIP/                          # Auto-created for inference output (WIP.yuv)
└── super_resolution_build/
    ├── ESPCN/                    # PyTorch TorchScript models (.pt)
    │   ├── epoch_2_100.pt
    │   ├── epoch_3_100.pt
    │   ├── epoch_4_100.pt
    │   └── epoch_8_100.pt
    └── EVASR/
        └── saliency_weight/      # Saliency weight matrices (.npy)
            └── 0.001_Beauty_m1_res.npy
```

## 5. Run Super-Resolution (`live_demo.py`)

### Headless Run

```bash
python live_demo.py Beauty --device cuda --scale 4 --backend none
```

### Live Playback Display (FFplay)

```bash
python live_demo.py Beauty --device cuda --scale 4 --backend display -v
```

### Output is saved to `WIP/WIP.yuv` for VMAF Evaluation

```bash
python live_demo.py Beauty --device cuda --scale 4 --backend eval
```

## 6. Run VMAF CUDA Evaluation (`live_demo.sh`)

Ensure the script is executable before running:

```bash
chmod +x live_demo.sh
```

### Usage

```bash
sudo ./live_demo.sh
```

## 7. Citation

Zichen Zhu, Stefano Petrangeli, Yao Liu, and Sheng Wei.
*Computation-Adaptive Video Streaming.*
IEEE International Workshop on Multimedia Signal Processing (MMSP), September 2026.

## License

MIT License
