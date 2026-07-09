# Error Log — Fusion Project

> All errors must be logged here for team reference. Do not fix bugs silently.

| Date | File | Error | Cause | Fixed By | Status |
|------|------|-------|-------|----------|--------|
| 2026-07-09 | core/types.py | `ImportError: cannot import 'Modality' from 'fusion.constants'` | `fusion.constants` module with `Modality` enum had not been created yet | Lead Dev — created `constants.py` with `Modality(str, Enum)` (VISION/LANGUAGE/AUDIO) | Fixed |
| 2026-07-09 | core/alignment.py | `ImportError: cannot import 'Modality' from 'fusion.constants'` | Same missing `fusion.constants` dependency | Lead Dev — same fix; also created `exceptions.py` (9 classes) and `core/modal_tensor.py`, `core/schema.py` | Fixed |
| 2026-07-09 | constants.py | Breaking change: `Modality` enum members renamed | Previous members (`TEXT`, `IMAGE`, `AUDIO`, `VIDEO`, `TABULAR`, `POINT_CLOUD`) replaced with (`VISION`, `LANGUAGE`, `AUDIO`) | Lead Dev — intentional redesign per updated spec | Fixed |
| 2026-07-10 | README.md | `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0` during `pip install -e .` | `README.md` was encoded as UTF-16 LE with BOM (`ff fe`); setuptools `file: README.md` directive expects UTF-8 | Lead Dev — re-encoded file to UTF-8 via `iconv` and stripped `\r\n` to `\n` | Fixed |

