# Windows development setup (WSL2 guide retired)

This project now targets **native Windows development** using the user's existing Python environment.

The former WSL2 setup instructions are intentionally retired. Do not install WSL2 or create a new Python virtual environment as part of the default workflow.

## Existing environment

From PowerShell, run commands in the existing PopCornConvVoice checkout and use the Python interpreter already used by the project:

```powershell
python --version
python -m pip --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

These are environment diagnostics only. They are not an AudioDec reconstruction test.

## External AudioDec checkout

The adapter expects the AudioDec repository root to contain:

- `utils/audiodec.py`
- `bin/stream.py`
- `models/`
- The selected pretrained encoder checkpoint and its adjacent `config.yml`

Set the checkout and checkpoint paths when constructing `AudioDecOneToOneConverter`. Do not copy AudioDec source into this repository.

## Project tests

Run only PopCornConvVoice-owned tests:

```powershell
python -m pytest
```

The adapter tests use mocked AudioDec components and do not load AudioDec checkpoints or run its standalone demo.
