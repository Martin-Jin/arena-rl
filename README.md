# Arena AI: A Fighter That Learns From You

A 2D arena fighting game where the opponent is a reinforcement learning (RL) agent. It is first pretrained by fighting copies of itself, then fine-tuned on how **you** play. Every time you beat it, the fight becomes training data, so it adapts to your habits and you have to keep changing your strategy.

This repository contains the **Python side only**: the game simulation, training code, and a pygame viewer. A port to Unity/Godot (using ONNX-exported networks) is a possible later step and is out of scope here.

> Status: planning / early development

---

## Core idea

1. **Self-play pretraining:** two untrained agents fight each other until they are competent but generic.
2. **Play against it:** you fight the agent. Every frame (observation, action, reward) is logged.
3. **Learn between rounds:** the agent trains on your recorded fights, with a visible learning phase (loss graph, win rate, heatmaps).
4. **Repeat:** the agent gets harder as it learns your habits.

## Planned features

- Discrete-action 2D fighter (move, jump, attack, block, dodge)
- Gymnasium-compatible environment, headless mode for fast training
- Self-play with a league of past checkpoints (also serves as a "ghost league" of opponents from novice to expert)
- Behaviour-cloning model of the human player, used to run many fast simulated fights between rounds
- Learning UI: loss, win-rate graphs, hit heatmaps
- Tunable learning rate and learning budget per round (difficulty control)

## Tech stack

|Purpose|Tool|
|---|---|
|Language|Python 3.10+|
|Environment API|Gymnasium, PettingZoo (multi-agent)|
|Networks / training|PyTorch, Stable-Baselines3 (baseline), own PPO (later)|
|Rendering / input|pygame|
|Logging / plots|TensorBoard or Weights & Biases, matplotlib|
|Export (future)|ONNX|

## Suggested repo layout

```
arena-ai/
├── arena/              # game simulation (pure Python, no rendering)
│   ├── env.py          # Gymnasium/PettingZoo environment
│   ├── fighter.py      # movement, attacks, cooldowns, damage
│   └── spec.md         # observation/action/reward spec (porting checklist)
├── agents/
│   ├── scripted.py     # hand-coded bots (baselines)
│   ├── ppo.py          # own PPO implementation (later phase)
│   └── clone.py        # behaviour-cloning model of the player
├── training/
│   ├── selfplay.py     # self-play + opponent pool
│   └── finetune.py     # between-round training
├── viewer/             # pygame viewer and human-play mode
├── data/               # logged fights (gitignored)
├── checkpoints/        # saved models (gitignored)
├── notebooks/          # experiments and analysis
└── README.md
```

## Design principles

- **Deterministic, fixed timestep.** No dependence on engine-specific physics quirks, which keeps a later port feasible.
- **Simple custom collision** (boxes and velocities) instead of a full physics engine.
- **Write the spec as you go.** Keep observation order, scaling, frame rate, action set, and reward in `arena/spec.md`.
- **Log everything.** Record fights so behaviour can be replayed and compared across versions.
- **One change at a time** when experimenting, and log the result.

## High-level plan (~80 hours)

|Phase|Goal|Est. hours|
|---|---|---|
|1. Environment|Arena, fighter, manual controls, Gymnasium wrapper, headless mode|12|
|2. Baselines and self-play|Scripted bots, PPO self-play, opponent pool|15|
|3. Player modelling|Fight logging, behaviour-cloning model of you|12|
|4. Adaptation|Between-round fine-tuning against your clone, replay buffer|15|
|5. Interface|Learning UI, heatmaps, ghost league|10|
|6. Evaluation|Tuning, test with other people, write-up|16|

Stretch goals: implement PPO from scratch and compare with Stable-Baselines3; add limb physics with PD-controlled joints; export to ONNX and port to Unity or Godot.

## Known risks

- **Cold start:** random agents rarely land hits. Use small shaping rewards early, then remove them.
- **Cycling and degenerate strategies:** use an opponent pool of past versions; watch for agents that hide or stand still.
- **Overfitting to one player:** test against someone with different habits.
- **Forgetting:** mix older data into fine-tuning.
- **Non-stationarity:** an opponent that keeps changing makes training unstable, so log win rate against fixed past checkpoints.

## Starting resources

**Reinforcement learning fundamentals**

- [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course): practical, beginner friendly
- [OpenAI Spinning Up in Deep RL](https://spinningup.openai.com/): concepts and key algorithms (read the PPO page)
- [Sutton & Barto, _Reinforcement Learning: An Introduction_](http://incompleteideas.net/book/the-book-2nd.html): free textbook, read chapters 1-6 and 13 first
- [Lilian Weng: Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)

**Tools and documentation**

- [Gymnasium docs](https://gymnasium.farama.org/): especially "Make your own custom environment"
- [Stable-Baselines3 docs](https://stable-baselines3.readthedocs.io/)
- [PettingZoo docs](https://pettingzoo.farama.org/): multi-agent environments
- [PyTorch tutorials](https://pytorch.org/tutorials/)
- [pygame docs](https://www.pygame.org/docs/)

**Implementing PPO yourself**

- [The 37 Implementation Details of PPO (ICLR blog track)](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- [CleanRL](https://github.com/vwxyzjn/cleanrl): single-file, readable reference implementations

**Self-play and multi-agent papers**

- Bansal et al., _Emergent Complexity via Multi-Agent Competition_ (arXiv 1710.03748)
- Silver et al., _Mastering Chess and Shogi by Self-Play with a General RL Algorithm_ (AlphaZero, arXiv 1712.01815)
- OpenAI, _Emergent Tool Use from Multi-Agent Autocurricula_ (hide-and-seek)

**Behaviour cloning / imitation learning**

- Search for "behavioral cloning" in the Hugging Face Deep RL Course, and the `imitation` library docs

## Getting started

```bash
git clone <your-repo-url>
cd arena-ai
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1` instead of the `source` command.

The editable install reads the dependencies from `pyproject.toml` and installs
the libraries currently required by the code: PyTorch, pygame-ce, and PyYAML.
The `pygame-ce` package provides the `pygame` module used by the viewer.

The following packages are planned for later environment and training work and
are not required by the current implementation:
`gymnasium`, `pettingzoo`, `stable-baselines3`, `numpy`, `matplotlib`, and
`tensorboard`.

Suggested first steps:

1. Work through the first units of the Hugging Face Deep RL Course.
2. Build a minimal one-fighter environment with the Gymnasium API and test it with random actions.
3. Train a PPO agent against a simple scripted bot using Stable-Baselines3.
4. Add the second agent and move to self-play.

## Roadmap

- [ ] Phase 1: environment and manual play
- [ ] Phase 2: scripted bots and self-play
- [ ] Phase 3: fight logging and player model
- [ ] Phase 4: between-round adaptation
- [ ] Phase 5: learning UI and ghost league
- [ ] Phase 6: evaluation and write-up
- [ ] Stretch: own PPO, limb physics, ONNX export / Unity port

## License

Choose one (e.g. MIT) and add a `LICENSE` file.