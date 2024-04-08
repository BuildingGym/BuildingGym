# BuildingGym
![pypi](https://pypi-camo.freetls.fastly.net/49eddcb8b6dd234c39f1459da9dcc601043be0a7/68747470733a2f2f696d672e736869656c64732e696f2f707970692f762f77616e6462)  ![conda](https://camo.githubusercontent.com/ef8ab0467fad6b69d198eddeae5e64463478a8e901683abfebfbf583ca8ba3b5/68747470733a2f2f696d672e736869656c64732e696f2f636f6e64612f766e2f636f6e64612d666f7267652f77616e6462)

<img src="docs\README_images\pytorch_logo.png" width="20%" >


[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)
## Introduction
![image](Figures//BuildingGym.png)
BuildingGym is a project that provides an API to easily train reinforcement learning control algorithm for all EnergyPlus envionment, and includes implementations of common reinforcement learning algorithm: Policy gradient, DQN, A2C, A3C, and more. Below is the structure for ```BuildingGym```

## Features
- 😏  Applied to all user-defined Energyplus model
- ❤️ Easy implement for common RL algorithms
- 💥  Included commone RL algorithm
- 😆 Auto-select the best model
- 😋 Track and visualize all the training process
- 😃 Applied to common control problem, e.g. demand respond, energy saving etc.

## Preparation
Install [Energyplus](https://energyplus.net/), [Pytorch](https://pytorch.org/), [Gymnasium](https://github.com/Farama-Foundation/Gymnasium), [Wandb](https://wandb.ai/site)
## Quickstart
### Read idf
Read the Energyplus model (```.idf file```)
```
a = 1
```


## Methods in IDF class
