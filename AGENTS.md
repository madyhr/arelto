# Arelto Codebase Guide for Agents

## Project Overview
Arelto (RL2) is a Reinforcement Learning Rogue-Lite game mixing C++ (engine/performance) and Python (RL logic/tooling).
- **Core Engine:** C++20 (SDL2, OpenMP)
- **RL/Bindings:** Python 3.10+, PyTorch, pybind11

## Build & Environment

### Prerequisites
- Linux (Debian/Ubuntu/Arch)
- Python 3.10+
- C++ Compiler (C++20 support)
- CMake >= 3.10, Ninja
- SDL2, SDL2_image, SDL2_ttf, SDL2_mixer
- NVIDIA GPU (for async training)

### Setup Commands
```bash
# 1. Create venv
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies & build Python package
# Option A: pip (simplest)
pip install .

# Option B: CMake (development/fast rebuilds)
cmake -B build -G Ninja -DCMAKE_INSTALL_PREFIX=.
ninja -C build install
export PYTHONPATH=$PWD  # Often needed for local dev
```

### Running the Game
```bash
python scripts/start_game_async.py  # Standard mode (async RL)
python scripts/start_game.py        # Sync mode (debug/slow)
```

## Testing & Verification

### Python Tests (`tests/`)
```bash
# Run all tests
python -m pytest tests

# Run a specific test file
python -m pytest tests/test_specific_file.py

# Run a specific test case
python -m pytest tests/test_file.py::test_function_name
```

### C++ Tests (`tests/cpp/`)
Requires CMake build.
```bash
# Build tests
ninja -C build arelto_tests

# Run all C++ tests
./build/arelto_tests

# Run specific C++ test (GoogleTest filter)
./build/arelto_tests --gtest_filter=TestSuiteName.TestName
```

## Code Style & Conventions

### Python
- **Formatter:** Follow `black` style (88 line length).
- **Imports:** Grouped: Standard Lib -> Third Party -> Local (`rl`). Alphabetical.
- **Type Hints:** REQUIRED for function signatures.
  ```python
  def calculate_reward(state: torch.Tensor) -> float: ...
  ```
- **Naming:**
  - Variables/Functions: `snake_case`
  - Classes: `PascalCase`
  - Constants: `SCREAMING_SNAKE_CASE`
- **Path Handling:** Use `pathlib` or `os.path`. explicit absolute paths preferred in scripts.

### C++
- **Formatter:** `clang-format` (Google Style, 2-space indent, 80 col limit).
  - Config: `.clang-format` in root.
- **Standard:** C++20.
- **Naming:**
  - Classes/Methods: `PascalCase` (`Game::Initialize`, `Player`)
  - Variables: `snake_case` (`player_health`)
  - Private Members: trailing underscore `member_var_` (common Google style) or `m_memberVar` (check existing files).
  - Files: `snake_case.cpp` / `snake_case.h`
- **Memory:** Prefer `std::unique_ptr` / `std::shared_ptr` over raw pointers.
- **Headers:** `#pragma once` preferred if used, or standard include guards.
- **Namespace:** Wrap core logic in `namespace arelto { ... }`.

### Linting
- **C++:** `clang-tidy` (if available), `clang-format`
- **Python:** `ruff` or `flake8` (if configured), `mypy` for types.

## Directory Structure
- `src/`: C++ Engine source.
- `include/`: C++ Headers.
- `rl/`: Python RL package (PyTorch models, PPO algo).
- `scripts/`: Entry points.
- `tests/`: Unit tests.
- `extern/`: Submodules (pybind11, gtest).

## Rules for Agents
1. **Safety:** Do not commit secrets. Verify tests pass before asking user to commit.
2. **Context:** Read related files before editing. Check `CMakeLists.txt` when adding C++ files.
3. **Types:** Always add type hints when writing new Python code.
4. **Performance:** Be mindful of the C++/Python boundary. Minimize data copying across pybind11.
