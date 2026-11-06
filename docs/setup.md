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

Install the project in editable mode:

```powershell
py -m pip install --upgrade pip
py -m pip install -e .
```

The project dependencies are declared in [`pyproject.toml`](../pyproject.toml).
The editable install currently provides pygame-ce, PyYAML, and PyTorch.

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

## Verify the checkout

Run the current repository correctness check:

```powershell
py -m compileall assets src tests
```
