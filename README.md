# BuildingGym
![pypi](https://pypi-camo.freetls.fastly.net/49eddcb8b6dd234c39f1459da9dcc601043be0a7/68747470733a2f2f696d672e736869656c64732e696f2f707970692f762f77616e6462) ![License](https://img.shields.io/badge/license-MIT-blue)

<img src="docs\README_images\pytorch_logo.png" width="10%" > <img src="https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/gymnasium-text.png" width="13%" >

> ## 📢 Announcement  
> We keep updating our project🚀.
> If you like this project, consider giving it a **star** 🌟 and clicking **watch** 👀 to keep up with the latest updates and improvements. Your support means a lot!

🎉 **News:** Our paper, "[BuildingGym: An open-source toolbox for AI-based building energy management using reinforcement learning](https://link.springer.com/article/10.1007/s12273-025-1306-y)", has been recognized as an **ESI Highly Cited Paper** and selected as the **Cover Article** in *Building Simulation*.
<p align="center">
<img src="docs\README_images\cover_paper.png" width="38%" >
</p>
## Introduction
🔥 BuildingGym is a project that provides an API to easily train reinforcement learning control algorithm for all EnergyPlus environment, and includes implementations of common reinforcement learning algorithm: Policy gradient, DQN, A2C, A3C, and more. Below is the structure for ```BuildingGym```

🎓 BuildingGym is free for students, educators, and academic researchers.

<!-- <p align="center"> -->
<!-- <img src="docs\README_images\Package structure.png" width="66%" > -->
<!-- </p> -->

## Features
- 😏 Applied to all user-defined EnergyPlus model
- ❤️ Easy implement for common RL algorithms
- 💥 Include common RL algorithms
- 😆 Auto-select the best model
- 😋 Track and visualize all the training process
- 😃 Applied to common control problem, e.g. demand respond, energy saving etc.

## Preparation
Please install this package with python=3.11 and follow Installation guide.docx. 

Please note the supported EnergyPlus model file is v23.2. For model in other version, please convert to 23.2. See [here](https://bigladdersoftware.com/epx/docs/9-4/tips-and-tricks-using-energyplus/converting-older-version-energyplus-files.html) for help

## License
[MIT license](https://github.com/wandb/wandb/blob/main/LICENSE)

## Acknowledgement
The EnergyPlus model in showcase refers to [Large Office](https://bitbucket.org/sbslab-zuo/bem-cbecs/get/LargeOffice.zip) model in project of [Building Energy Models for Commercial Buildings Based on CBECS Data](https://www.colorado.edu/lab/sbs/BEM)

## Citation
Please cite our paper:
<pre><code class="language-bibtex">
@article{dai2025buildinggym,
  title={BuildingGym: An open-source toolbox for AI-based building energy management using reinforcement learning},
  author={Dai, Xilei and Chen, Ruotian and Guan, Songze and Li, Wen-Tai and Yuen, Chau},
  Journal={Building Simulation},
  pages={1--19},
  year={2025},
  organization={Springer}
}
 </code></pre>
