# Development setup

## Requirements

- Python 3.10 or newer
- The Windows `py` launcher, or an equivalent Python command on another
  platform

## Install

From the repository root, choose one of the three options below. The project
dependencies are declared in [`pyproject.toml`](../pyproject.toml). The project
install provides NumPy, pygame-ce, and PyYAML.

The `--upgrade-deps` flag in the `venv` commands installs the latest pip into the
new environment; remove it if you don't want that.

### Option A: CPU only

Create and activate a virtual environment:

```powershell
py -m venv --upgrade-deps .venv
.\.venv\Scripts\Activate.ps1
```

Install the project with its CPU extra, which also installs the standard PyPI
PyTorch wheel:

```powershell
py -m pip install -e ".[cpu]"
```

### Option B: NVIDIA CUDA, reusing a global PyTorch install

Use this to avoid downloading a multi-GB CUDA PyTorch into every virtual
environment. PyTorch is installed once in your global Python, and the virtual
environment is created with access to global packages.

If the CUDA build of PyTorch is not installed globally yet, install it once from
outside any virtual environment (run `deactivate` first if one is active):

```powershell
py -m pip install --upgrade pip
py -m pip install torch --index-url https://download.pytorch.org/whl/cu132
```

Then create the virtual environment with access to global packages and install
the project:

```powershell
py -m venv --system-site-packages --upgrade-deps .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -e .
```

Do not install the `cpu` extra in this environment, because it requests the CPU
PyTorch wheel and would shadow the global CUDA build.

To recreate an existing environment, run `deactivate`, delete `.venv`
(`Remove-Item -Recurse -Force .venv`), then repeat the commands above. Deleting
`.venv` does not affect the global PyTorch install.

### Option C: NVIDIA CUDA, installed inside the virtual environment

Install the PyTorch wheel built for CUDA before installing the project. This
prevents pip from selecting the CPU wheel:

```powershell
py -m venv --upgrade-deps .venv
.\.venv\Scripts\Activate.ps1
py -m pip install torch --index-url https://download.pytorch.org/whl/cu132
py -m pip install -e .
```

The `cu132` index is an example in options B and C. Choose the CUDA wheel index
matching the PyTorch and NVIDIA driver versions supported by your machine using
the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).
Do not install the `cpu` extra in this environment.

### Verify PyTorch and the GPU

With the virtual environment active:

```powershell
py -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

For CUDA setups, the second line should print `True`. If it prints `False` in
option B, the global PyTorch is a CPU build or the virtual environment is not
using it.

### PowerShell activation errors

If PowerShell prevents activation scripts from running, allow scripts for the
current PowerShell process only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## Verify the checkout

Run the compile check and the unit tests:

```powershell
py -m compileall src tests scripts
py -m unittest discover -s tests
```

## Deactivate and delete the environment

Leave the active virtual environment:

```powershell
deactivate
```

Delete it (this removes every package installed in it, including the editable
install of this project, but not global packages or other environments):

```powershell
Remove-Item -Recurse -Force .venv
```

Then recreate it with the commands in [Install](#install).