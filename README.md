Lunar Lander
============

A reinforcement learning agent for [OpenAI Gym](https://gym.openai.com/envs/LunarLander-v2).

![Landing](images/landing.gif)

## How to Run

1. Install Python v2.7+
2. (Optional) Invoke a virtual environment via `virtualenv`
3. Install dependencies by running `pip install -r requirements.txt`
4. Run `./run` to reproduce experiments (graphs may differ due to randomness)

> NOTE: If Box2D is causing errors, you may need to build from source. Run `git
> submodule init && git submodule update` and `./build_pybox2d`.

## References

- Bill Learning's YouTube video, https://www.youtube.com/watch?v=Lv_VDz1RhWY
- Fiszel, Reinforcement Learning and DQN, https://rubenfiszel.github.io/posts/rl4j/2016-08-24-Reinforcement-Learning-and-DQN.html
- Hasselt et al, Deep Reinforcement Learning with Double Q-Learning (2015), https://arxiv.org/pdf/1509.06461.pdf
- Matiisen, Demystifying Deep Reinforcement Learning (2015), https://www.intelnervana.com/demystifying-deep-reinforcement-learning/
- Minh et al, Playing Atari with Deep Reinforcement Learning (2015), https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
