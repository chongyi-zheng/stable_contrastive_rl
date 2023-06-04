# [Stabilizing Contrastive RL: Techniques for Offline Goal Reaching](arxiv_link)
<p align="center"><img src="stable_contrastive_rl.png" width=90%></p>

<p align="center"> Chongyi Zheng, &nbsp; Benjamin Eysenbach, &nbsp; Homer Walke, &nbsp; Patrick Yin, &nbsp; Kuan Fang, <br> Ruslan Salakhutdinov &nbsp; Sergey Levine</p>
<p align="center">
   Paper: <a href="arxiv_link">arxiv_link</a>
</p>
<p align="center">
   Website: <a href="https://chongyi-zheng.github.io/stable_contrastive_rl">https://chongyi-zheng.github.io/stable_contrastive_rl</a>
</p>

*Abstract*: In the same way that the computer vision (CV) and natural language processing (NLP) communities have developed self-supervised methods, reinforcement learning (RL) can be cast as a self-supervised problem: learning to reach any goal, without requiring human-specified rewards or labels. However, actually building a self-supervised foundation for RL faces some important challenges. Building on prior contrastive approaches to this RL problem, we conduct careful ablation experiments and discover that a shallow and wide architecture, combined with careful weight initialization and data augmentation, can significantly boost the performance of these contrastive RL approaches on challenging simulated benchmarks. Additionally, we demonstrate that, with these design decisions, contrastive approaches can solve real-world robotic manipulation tasks, with tasks being specified by a single goal image provided after training.

This repository contains code for running stable contrastive RL algorithm.

```
TODO: BibTex
```

## Installation

### Create Conda Env

Install and use the included anaconda environment.
```
$ conda env create -f conda_env/stable_contrastive_rl.yml
$ source activate stable_contrastive_rl
(stable_contrastive_rl) $
```

### Dependencies

Download the dependency repos.
- [bullet-manipulation](https://github.com/chongyi-zheng/bullet-manipulation) (contains environments): ```https://github.com/chongyi-zheng/bullet-manipulation```
- [multiworld](https://github.com/vitchyr/multiworld) (contains environments): ```git clone https://github.com/vitchyr/multiworld```

Add dependency paths.
```
export PYTHONPATH=$PYTHONPATH:/path/to/multiworld
export PYTHONPATH=$PYTHONPATH:/path/to/bullet-manipulation
export PYTHONPATH=$PYTHONPATH:/path/to/bullet-manipulation/bullet-manipulation/roboverse/envs/assets/bullet-objects
```

### Setup Config File

You must setup the config file for launching experiments, providing paths to your code and data directories. Inside `railrl/config/launcher_config.py`, fill in the appropriate paths. You can use `railrl/config/launcher_config_template.py` as an example reference.

```cp railrl/launchers/config-template.py railrl/launchers/config.py```

## Running Experiments

### Offline Dataset and Goals
Download the simulation data and goals from [here](). Alternatively, you can recollect a new dataset by following the instructions in [bullet-manipulation](https://github.com/chongyi-zheng/bullet-manipulation).



## Questions?
If you have any questions, comments, or suggestions, please reach out to Chongyi Zheng (chongyiz@andrew.cmu.edu).