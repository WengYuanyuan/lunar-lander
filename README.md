Lunar Lander
============

A reinforcement learning agent for [OpenAI Gym](https://gym.openai.com/envs/LunarLander-v2).

![Landing](images/landing.gif)

## How to Run

1. Install Python v2.7+
2. (Optional) Invoke a virtual environment via `virtualenv`
3. Install dependencies by running `pip install -r requirements.txt`
4. Run `python run.py` to reproduce all experiments, or...
5. Run `python plot.py` to plot saved data

> NOTE: If Box2D is causing errors, you may need to build from source. Run `git
> submodule update --recursive --remote` and `./build_pybox2d`.
