# Arelto

A Reinforcement Learning Rogue-Lite (RL2) game where the enemies get smarter over time. Try to survive for as long as you can!

## Prerequisites

Before installing, ensure you have the following system dependencies installed.

### Linux
#### Debian/Ubuntu
```bash
sudo apt update
sudo apt install cmake build-essential python3-dev
sudo apt install libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev
sudo apt install libomp-dev
```

#### Arch
```bash
sudo pacman -S cmake base-devel python sdl2 sdl2_image sdl2_ttf
```

![NOTE]: Windows and macOS are currently not supported.

### Python
- Python 3.10 or higher.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/arelto.git
    cd arelto
    ```

2. **Fetch external dependencies:**
    ```bash
    git submodule update --init --recursive
    ```

3.  **Create a virtual environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **Install the game as a package:**
    ```bash
    pip install .
    ```
    *Note: This process may take some time as it downloads all dependencies and compiles the C++ backend.*

## Running the Game

To start the game with the asynchronous agent (default experience):

```bash
python scripts/start_game_async.py
```

Alternatively, there is also a synchronous agent variant. However, the game pauses during policy updates, so it is also not as smooth an experience.

```bash
python scripts/start_game.py
```

## Development

### Building
For development, it is recommended that you use an editable pip install:

```bash
pip install -e . -v
```

This will automatically handle CMake configuration and build steps using `scikit-build-core`.

### Tests
- **Running Tests:**
    #### Python tests
    ```bash
    python -m pytest tests
    ```

    #### C++ tests
    ```bash
    ./build/arelto_tests
    ```
