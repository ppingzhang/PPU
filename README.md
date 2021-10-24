# PPU: Progressive Point Cloud Upsampling via Differentiable Rendering
by Pingping Zhang, Xu Wang, Lin Ma, Shiqi Wang, SamKwong, Jianmin Jiang

### Introduction 

This repository is for our TCSVT 2021 paper '[Progressive Point Cloud Upsampling via Differentiable Rendering](https://github.com/ppingzhang/PPU.git)'. 


### Installation
This repository is based on Tensorflow and the TF operators from PointNet++. Therefore, you need to install tensorflow and compile the TF operators. 

For installing tensorflow, please follow the official instructions in [here](https://www.tensorflow.org/install/install_linux). The code is tested under TF1.14 (higher version should also work) and Python 3.7 on Ubuntu 16.04.

### Usage

1. Clone the repository:

   ```shell
   https://github.com/ppingzhang/PPU.git
   cd PPU
   ```
   
2. Compile the TF operators
   For compiling TF operators, please check `tf_xxx_compile.sh` under each op subfolder in `code/tf_ops` folder. Or you can compile all by running "complie.sh". Note that you need to update `nvcc`, `python` and `tensoflow include library` if necessary. 
   
3. Train the model:
    First, you need to download the training patches in HDF5 format from [GoogleDrive](https://drive.google.com/file/d/1ZtWDvbeDTHcGTc1tjmPOI1Ydg54pty3w/view?usp=sharing) and put it in folder `data/train`.
    Then run:
   ```shell
   python3 mian.py --phase=train
   ```

4. Evaluate the model:
    First, you need to download the pretrained model from [GoogleDrive](https://drive.google.com/drive/folders/1IAndtnTSE2gw_2ugv73-_GR-CQGnV-S2?usp=sharing), extract it and put it in folder 'log'.
    Then run:
   ```shell
   python3 main.py --phase=test --log_dir=./log
   ```
   You will see the output results in the folder `./result/Ours/`.
   
5. The testing mesh files can be downloaded from [GoogleDrive](https://drive.google.com/drive/folders/1VvUPIIWERpeJudGvpXQhFNqK2q9jSa1I?usp=sharing).

### Evaluation code
We provide the code to calculate the uniform metric in the evaluation code folder. In order to use it, you need to install the CGAL library. Please refer [this link](https://www.cgal.org/download/linux.html) and  [PU-Net](https://github.com/yulequan/PU-Net) to install this library.
Then:
   ```shell
   cd evaluation_code
   cmake .
   make
   ./evaluation Icosahedron.off Icosahedron.xyz
   ```
The second argument is the mesh, and the third one is the predicted points.

## Citation

If Our model is useful for your research, please consider citing:
```shell
@article{zhang2021progressive,
  title={Progressive Point Cloud Upsampling via Differentiable Rendering},
  author={Zhang, Pingping and Wang, Xu and Ma, Lin and Wang, Shiqi and Kwong, Sam and Jiang, Jianmin},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2021},
  publisher={IEEE}
}
```
## Acknowledgement
Our code refers to the [PU-GAN](https://liruihui.github.io/publication/PU-GAN/)
 
