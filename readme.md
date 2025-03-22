This is the code implementation result of Team One Go Go in the NTIRE competition of CVPR 2025. 

Retrain Restormer for deraining task and fine-tune IPT for defocus-blurring task. We use two existing methods as follows:

1. Zamir S W, Arora A, Khan S, et al. Restormer: Efficient transformer for high-resolution image restoration[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022: 5728-5739.
2. Chen H, Wang Y, Guo T, et al. Pre-trained image processing transformer[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021: 12299-12310.

### Installation

This repository is built in PyTorch 1.10.2 and tested on Ubuntu 20.04 environment (Python3.8, CUDA11.3) with 4 3090 GPUs. Follow these instructions

1. Clone repository

   ```bash
   git clone https://github.com/tukumax/One-Go-Go.git
   ```

2. Create conda environment

   ```bash
   conda create -n ntire2025 python=3.8
   conda activate ntire2025
   ```

   

3. Install dependencies
   `torch 1.10.2+cu113, torchvision 0.11.3+cu113`

   ```bash
   conda install pytorch=1.10.2 torchvision cudatoolkit=11.3 -c pytorch
   pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
   pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
   # install basicsr
   python setup.py develop --no_cuda_ext
   ```

   

### Training

1. Raindrops removal

   ```bash
   ./train.sh Deraining/Options/Deraining_Restormer_RainDrop.yml
   ```

2. Defocus-blurring

   ```bash
   ./train.sh Defocus_Deblurring/Options/DefocusDeblur_Single_8bit_IPT_Blur.yml
   ```

### Testing

[Pretrained weights and submission images](https://pan.baidu.com/s/15KjE-ZVCF6LJZzGpBzPbxw?pwd=haha)

First, generate the images with raindrops removed, and then perform deblurring.

1. Raindrops removal

   ```bash
   python demo_Restormer.py --task Deraining --input_dir ./Deraining/Datasets/test/example --result_dir ./Defocus_Deblurring/Datasets/result
   ```

2. Defocus-blurring

   ```bash
   python demo_IPT.py --task Single_Image_Defocus_Deblurring --input_dir ./Defocus_Deblurring/Datasets/result/Deraining --result_dir ./Defocus_Deblurring/Datasets/result --tile 48 --tile_overlap 8
   ```

   