# Error Log — Fusion Project

> All errors must be logged here for team reference. Do not fix bugs silently.

| Date | File | Error | Cause | Fixed By | Status |
|------|------|-------|-------|----------|--------|
| 2026-07-09 | core/types.py | `ImportError: cannot import 'Modality' from 'fusion.constants'` | `fusion.constants` module with `Modality` enum has not been created yet | — | Open |
| 2026-07-09 | core/alignment.py | `ImportError: cannot import 'Modality' from 'fusion.constants'` | Same missing `fusion.constants` dependency; also imports `ModalTensor` from `fusion.core.modal_tensor` (exists) | — | Open |
