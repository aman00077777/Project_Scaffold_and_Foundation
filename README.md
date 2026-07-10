<p align="center">
  <h1 align="center">🔀 Fusion</h1>
  <p align="center">
    <strong>A modular multimodal fusion framework for deep learning</strong>
  </p>
</p>

---

> **Phase 1 — Project Scaffold and Foundation**
>
> Core abstractions, validation, and utilities are in place.
> Fusion strategy implementations, training loops, and CLI are planned for future phases.

---

## Overview

**Fusion** is a Python framework for building multimodal deep learning models. It provides a pluggable architecture for combining **vision**, **language**, and **audio** modalities through strategies like cross-attention, gated fusion, and Perceiver-style mechanisms.

**Python ≥ 3.9** · **PyTorch ≥ 2.0**

---

## What's Built (Phase 1)

- ✅ **ModalTensor** — tensor wrapper with modality metadata, device & detach support
- ✅ **ModalitySchema** — declarative shape & modality validation at runtime
- ✅ **ModalityAligner** — learnable projections into a shared embedding space
- ✅ **FusionComponent** — abstract base with `get_config()` / `from_config()` serialisation
- ✅ **Evaluation metrics** — accuracy, top-k, mAP, NDCG
- ✅ **Utilities** — logging, reproducibility seeds, parameter freeze/unfreeze, config I/O
- ✅ **Exception hierarchy** — `FusionError` tree with structured `details` payloads
- ✅ **Project tooling** — Black, isort, Flake8, Pylint, mypy, pre-commit, tox, pytest

### What's Next

- 🔲 Modality encoders & fusion strategy implementations
- 🔲 Training loop & data pipeline
- 🔲 CLI, deployment (ONNX / TorchScript)
- 🔲 Experiment tracking (W&B / MLflow)

---

## Installation

```bash
git clone https://github.com/your-org/fusion.git
cd fusion
pip install -e .

# Development
pip install -r requirements-dev.txt -r requirements-test.txt
pre-commit install
```

---

## Quick Start

```python
import torch
from fusion.constants import Modality, TaskType
from fusion.core.schema import ModalitySchema
from fusion.core.alignment import ModalityAligner

# Define what your pipeline expects
schema = ModalitySchema(
    input_modalities=[Modality.VISION, Modality.LANGUAGE],
    output_type=TaskType.CLASSIFICATION,
    expected_shapes={
        Modality.VISION:   (3, 224, 224),
        Modality.LANGUAGE: (512,),
    },
)

# Validate inputs
inputs = {
    Modality.VISION:   torch.randn(8, 3, 224, 224),
    Modality.LANGUAGE: torch.randn(8, 512),
}
schema.validate(inputs)  # ✓ passes or raises ModalityError

# Align to a shared embedding space
aligner = ModalityAligner(
    input_dims={Modality.VISION: 2048, Modality.LANGUAGE: 768},
    target_dim=256,
)
aligned = aligner({
    Modality.VISION:   torch.randn(8, 2048),
    Modality.LANGUAGE: torch.randn(8, 768),
})
# Both modalities → shape (8, 256)
```

---

## Project Structure

```
fusion/
├── constants.py          # Modality, FusionType, TaskType enums
├── exceptions.py         # FusionError hierarchy
├── core/
│   ├── types.py          # Type aliases
│   ├── base.py           # FusionComponent ABC
│   ├── modal_tensor.py   # ModalTensor dataclass
│   ├── alignment.py      # ModalityAligner
│   └── schema.py         # ModalitySchema validation
└── utils/
    ├── helpers.py         # Seeds, parameter utils
    ├── logging.py         # Logger factory
    ├── metrics.py         # Accuracy, mAP, NDCG
    ├── io.py              # Checkpoint I/O
    ├── config.py          # YAML config loader
    ├── device.py          # Device placement
    ├── download.py        # Asset download
    ├── registry.py        # Component registry
    └── visualization.py   # Plots & visualisation
```

---

## Team Contributions

| Contributor | Contributions |
|---|---|
| **Aman Sharma** | Core engine — `ModalTensor`, `ModalitySchema`, `ModalityAligner`, `FusionComponent` base class, exception hierarchy, type system |
| **Aman Hingawe** | Root configuration & core modules — `setup.cfg`, `pyproject.toml`, project initialisation, debug & error tracking |
| **Gauri Ninawe** | Root configurations — `requirements*.txt`, `VERSION`, build system setup |
| **Shantanu Warghane** | Root linting — `.flake8`, `.pylintrc`, `.coveragerc`, `pre-commit` config, project structure |
| **Arya Tiwari** | Utilities — evaluation metrics, helper functions, logging, I/O, download, visualisation, testing & coverage config |

---

## Testing

```bash
pytest                              # Run all tests
pytest --cov=fusion --cov-report=html   # With coverage
tox                                 # Multi-environment
```

---

## License

See the `LICENSE` file for details.

---

<p align="center">
  <sub>Built with ❤️ by the Fusion Team 3</sub>
</p>
