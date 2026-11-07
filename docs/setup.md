# Development setup

## Requirements

- Python 3.10 or newer
- The Windows `py` launcher, or an equivalent Python command on another
  platform

## Install

From the repository root, create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

For a CPU-only installation, install the project with its CPU extra:

```powershell
py -m pip install --upgrade pip
py -m pip install -e ".[cpu]"
```

The project dependencies are declared in [`pyproject.toml`](../pyproject.toml).
The CPU extra installs the standard PyPI PyTorch wheel.

### NVIDIA CUDA installation

For an NVIDIA GPU, install the PyTorch wheel built for CUDA before installing
the project. This prevents pip from selecting the CPU wheel:

```powershell
py -m pip install --upgrade pip
py -m pip install torch --index-url https://download.pytorch.org/whl/cu126
py -m pip install -e .
```

The `cu126` index is an example. Choose the CUDA wheel index matching the
PyTorch and NVIDIA driver versions supported by your machine using the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/).
Do not install the `cpu` extra in this environment, because it requests the
CPU PyTorch wheel.

Verify that PyTorch can see the GPU:

```powershell
py -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If PowerShell prevents activation scripts from running, allow scripts for the
current PowerShell process only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## Run the viewer

With the virtual environment active:

```powershell
py -m arena_ai.viewer.game
```

The viewer controls are documented in the [project README](../README.md).

The default viewer includes the scripted opponent. Set `VIEW_OPPONENT = False`
in [`src/arena_ai/viewer/game.py`](../src/arena_ai/viewer/game.py) to switch to
single-fighter practice mode.

To watch training rollouts, set `VISUALIZE_TRAINING = True` in
[`scripts/run_training.py`](../scripts/run_training.py) before running the
training command. It renders every fixed simulation step without a real-time
delay.

## Verify the checkout

Run the current repository correctness check:

```powershell
py -m compileall assets src tests
```
