## Megatron-Core Agent Playbook

This guide orients autonomous agents working in this repository. It mixes build recipes, lint/test tips, and coding norms so tasks can be completed safely without rediscovery.

### 1. Environment & Tooling
- **Python**: Target 3.12 (`.python-version`). Use the repo’s venv; avoid system packages.
- **Package manager**: Pip/uv backed by `pyproject.toml`. Default extra for dev: `pip install -e .[dev]` (adds TE, grouped GEMM, etc.).
- **Pre-commit**: `.pre-commit-config.yaml` enforces `black`, `isort`, `pylint`. Run manually via `pre-commit run -a` when editing core files.
- **CUDA/TE**: Some ops depend on `transformer_engine`, `nv-grouped-gemm`, etc. Unit tests mock these when unavailable, but prefer GPUs for high-fidelity runs.

### 2. Build / Test / Lint Commands
- **Install (dev)**: `pip install --no-build-isolation -e .[dev]` (ensures TE compatibility). For minimal CPU work, `pip install -e .`.
- **Format**: `black megatron/core tests/unit_tests` and `isort megatron/core tests/unit_tests` (100-col). Both configured via pyproject; black skips trailing commas, string normalization.
- **Lint**:
  - `ruff check megatron/core tests` (rule S506 plus docstyle disables).
  - `flake8 megatron/core tests` (max line 100, ignores E203/E501, etc.).
  - `pylint megatron.core` (only specific warnings enabled; `print` banned).
- **Type hints**: No pyright config; rely on MyPy ad hoc if needed.
- **Tests**:
  - All unit tests: `pytest` (config via `pyproject`).
  - Single test file: `pytest tests/unit_tests/transformer/moe/test_moe_layer.py -k <case>`.
  - Collect only: `pytest --collect-only tests/...`.
  - Coverage: `pytest --cov=megatron --cov-report=term-missing` (coverage config in pyproject).
  - Functional (long): `pytest tests/functional_tests -m 'not slow'` (expect GPU + data).
- **CI mimic**: `uv pip install -e .` plus `uv pip install -r requirements.txt` if scripts rely on extras.

### 3. Repo Conventions
- **Imports**: `isort` config (profile black, line 100, sections FUTURE/STDLIB/THIRDPARTY/FIRSTPARTY/LOCALFOLDER). Third-party list includes `transformer_engine`. Keep first-party under `megatron`. Relative imports discouraged in core packages.
- **Formatting**: `black` 24.4 rules, skip string normalization & magic trailing comma; spaces over tabs; docstrings Google style (`ruff` expects). Comments > code where logic is non-obvious; avoid trailing whitespace.
- **Docstrings**: Only for non-trivial classes/functions per ruff configuration (Google style). Tests exempt.
- **Typing**: Prefer explicit types for public APIs, config dataclasses, and dispatcher interfaces. Use `Optional[torch.Tensor]`, `Sequence`, etc. Avoid `Any` except patching legacy APIs.
- **Naming**: snake_case for functions/vars, PascalCase for classes, UPPER for constants. Config flags live in `TransformerConfig`. Keep new feature knobs near similar options and default in dataclass.
- **Error handling**: Raise `ValueError` for invalid configs; use `RuntimeError` for runtime GPU sync issues. Avoid bare `except`; log with `logger = logging.getLogger(__name__)`.
- **Logging**: Standard Python logging. Don’t use `print` (pylint blocks). For debug, prefer `logger.debug`.
- **CUDA sync**: Use helper utilities in `moe_utils` for streams/events; keep `torch.cuda.current_stream()` semantics.
- **Testing style**: Pytest with fixtures; avoid `unittest`. Use deterministic seeds for GPU tests; `pytest -k` recommended for targeted runs.
- **Git**: Don’t touch user changes (see system instructions). Use `apply_patch` for in-place edits. Follow existing commit style (imperative, short). Avoid committing unless user requests.

### 4. MoE / Transformer Specific Guidance
- **Config plumbing**: When adding new behavior, propagate through `TransformerConfig`, `MoELayer`, dispatchers, and derived modules. Provide defaults and guard with `if self.config.<flag>`.
- **Dispatchers**: `token_dispatcher.py` organizes AllGather/AlltoAll/Flex. Maintain cudagraph attr lists and CPU sync logic. When adding metadata, ensure attributes capture for graph scope.
- **Experts**: `experts.py` hosts GroupedMLP/TEGroupedMLP/SequentialMLP. Keep GPU/CPU init parity; use `_initialize_affine_weight_cpu/gpu`. For caching/offload features, integrate with `ProcessGroupCollection` semantics.
- **Router**: `router.py` defines TopK router. Keep load balancing options (`moe_router_*`) there.
- **Rank sorter**: new module `rank_sorter.py`. When introducing new sorters, expose config knob `moe_rank_sorter` and builder callback.
- **Testing MoE**: tests in `tests/unit_tests/transformer/moe`. Use DP=2 setups by mocking `ProcessGroupCollection`. When verifying dispatch metadata, prefer CPU tensors for asserts.

### 5. Docs & Comments
- `docs/` uses Markdown; follow 80-120 columns for readability. For new features, update relevant doc (user-guide, API). Use fenced code blocks with language tags.
- `README.md` under `/workspace` is entry point; keep messaging aligned with NVIDIA guidelines.
- Mention new configuration knobs in docs and optionally `CHANGELOG.md` if user requests.

### 6. Handling External Tools
- No Cursor/Copilot instruction files present; default to this document plus repo configs.
- GPU-specific libs (TE, DeepEP, HybridEP) may be optional. Guard imports with `try/except ImportError` and set `HAVE_TE`, etc.
- When referencing TE-specific modules, wrap with availability checks and clear error messages.

### 7. Workflow Tips
- Before editing, run `git status -sb` to note dirty files (don’t revert user changes).
- Prefer `Read` tool for file inspection, `apply_patch` for editing. Avoid `cat`/`sed` via shell per system guidance.
- Keep responses concise, referencing files via inline code paths (e.g., `megatron/core/transformer/moe/moe_layer.py`).
- When implementing features:
  1. Update config and modules.
  2. Add/adjust tests.
  3. Run targeted pytest commands.
  4. Summaries should mention logic + verification steps.

### 8. Running a Single Test Examples
- **Specific file**: `pytest tests/unit_tests/transformer/moe/test_token_dispatcher.py`.
- **Specific test**: `pytest tests/unit_tests/transformer/moe/test_token_dispatcher.py -k test_alltoall_metadata`.
- **Debugging**: `pytest -vv --maxfail=1` for verbose first failure.
- **Distributed**: For multi-GPU, wrap with `torchrun --nproc_per_node=2 -m pytest tests/...` (only when test expects distributed init).

### 9. Performance & Memory
- Prefer tensor operations over Python loops. Use `torch.cuda.Stream` when overlapping CPU/GPU copies (see dispatcher code).
- When adding caches (e.g., expert weight cache), store pinned CPU buffers when possible and avoid per-step allocations.
- For large models, guard optional features behind config flags to avoid overhead in default path.

### 10. Documentation for Agents
- Keep this file updated whenever workflows or tooling changes (e.g., new lint rules, test entrypoints).
- When new directories are added (e.g., `/tools` scripts), document run instructions here or link to README.

*Stay aligned with NVIDIA coding standards, respect system instructions, and always run the narrowest test suite that exercises your changes before signaling completion.*
