# [Two by Two✌️ : Learning Multi-Task Pairwise Objects Assembly for Generalizable Robot Manipulation](https://tea-lab.github.io/TwoByTwo/)

<a href="https://tea-lab.github.io/TwoByTwo/"><strong>Project Page</strong></a>
  |
  <a href="https://arxiv.org/abs/2504.06961"><strong>arXiv</strong></a>
  |
  <a href=""><strong>Twitter</strong></a> 
  | <a href="https://docs.google.com/forms/d/e/1FAIpQLSfhPcAdky8ZojjPlSSHN4ubYqc7WHIwfiqFW2L5YpqAHbbVgg/viewform"><strong>Dataset</strong></a>

  <a href="https://scholar.google.com.hk/citations?hl=zh-CN&user=UZSbtlsAAAAJ">Yu Qi*</a>, 
  <a href="https://scholar.google.com.hk/citations?user=jOPXmhIAAAAJ&hl=zh-CN">Yuanchen Ju*</a>, 
  <a href="https://www.stillwtm.site/">Tianming Wei</a>, 
  <a href="https://cc299792458.github.io/">Chi Chu</a>, 
  <a href="https://www.ccs.neu.edu/home/lsw/">Lawson L.S. Wong</a>, 
  <a href="http://hxu.rocks/">Huazhe Xu</a>


**CVPR, 2025**


<div align="center">
  <img src="assets/teaser.png" alt="2by2" width="100%">
</div>


# 🛠️ Installation

This project is tested on Ubuntu 22.04 with CUDA 11.8. 

- Install [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install#macos-linux-installation:to-download-a-different-version) or [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer)
- Clone the repository and create the environment. The environment should be installed correctly within minutes. 

```python
git clone git@github.com:TEA-Lab/TwoByTwo.git
conda env create -f environment.yml
conda activate 2by2
```

- (Optional) If you would like to calculate Chamfer Distance, clone the [CUDA-accelerated Chamfer Distance library](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch/tree/master):

```python
cd src/shape_assembly/models
git clone https://github.com/ThibaultGROUEIX/ChamferDistancePytorch.git
```

## 🧩 Dataset

2BY2 Dataset has been released. To obtain our dataset, please fill out this [form](https://docs.google.com/forms/d/e/1FAIpQLSfhPcAdky8ZojjPlSSHN4ubYqc7WHIwfiqFW2L5YpqAHbbVgg/viewform?usp=sharing).

## 🍰 Dataset Utility Support
It is recommended to use our pre-generated point cloud. In the meantime, you can also generate your own point cloud, add your own data, or generate **URDF(Unified Robot Description Format)** file for robot simulation purpose, please see `data_util` folder for more detailed instructions.

## 🐰Training and Inference

In `src/config` modify the path of `log_dir` `data root_dir`. We support Distributed Data Parallel Training.


- Train Network B

```python
cd src
python script/our_train_B.py --cfg_file train_B.yml
```

- Train Network A
```python
cd src
python script/our_train_A.py --cfg_file train_A.yml
```
- Inference

```python
cd src
python script/our_eval.py
```

# 🎟️ Licence
This repository is released under the MIT license. Refer to [LICENSE](LICENSE) for more information.

# 🎨 Acknowledgement & Contact
Our codebase is developed based on [SE3-part-assembly](https://crtie.github.io/SE-3-part-assembly/), and we express our gratitude to all the authors for their generously open-sourced code, as well as the open-source contributions of all baseline projects [Puzzlefusion++](https://puzzlefusion-plusplus.github.io.), [Jigsaw](https://jiaxin-lu.github.io/Jigsaw/), [Neural Shape Mating](https://neural-shape-mating.github.io/). for their valuable impact on the community.

For inquiries about this project, please reach out to **Yu Qi: qi.yu2@northeastern.edu** and **Yuanchen Ju: juuycc0213@gmail.com**. You’re also welcome to open an issue or submit a pull request!😄


# 🎸 BibTeX

We would appreciate it if you find this work useful and consider citing it.
```
@article{qi2025two,
  title={Two by two: Learning multi-task pairwise objects assembly for generalizable robot manipulation},
  author={Qi, Yu and Ju, Yuanchen and Wei, Tianming and Chu, Chi and Wong, Lawson LS and Xu, Huazhe},
  journal={CVPR 2025},
  year={2025}
}
```
