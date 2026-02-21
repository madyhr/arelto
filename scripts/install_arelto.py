#!/usr/bin/env python3
import os
import platform
import shutil
import subprocess
import sys


def run_command(command_list, console=None, cwd=None, env=None, show_error=True):
    """
    Runs a shell command silently, returning the exit code.
    Expects command_list to be a list of strings to avoid shell injection and cross-platform path issues.
    """
    try:
        process = subprocess.run(
            command_list, cwd=cwd, env=env, capture_output=True, text=True
        )

        if process.returncode != 0 and show_error:
            # Safely try to use Rich if it has been bootstrapped
            if console:
                from rich.panel import Panel

                cmd_str = " ".join(command_list)
                console.print(f"\n[bold red]✗ Command failed:[/bold red] {cmd_str}")
                if process.stdout:
                    console.print(
                        Panel(
                            process.stdout,
                            title="Standard Output",
                            border_style="yellow",
                        )
                    )
                if process.stderr:
                    console.print(
                        Panel(
                            process.stderr, title="Standard Error", border_style="red"
                        )
                    )
            else:
                cmd_str = " ".join(command_list)
                print(f"\nERROR: Command failed: {cmd_str}")
                if process.stdout:
                    print(f"--- Standard Output ---\n{process.stdout}")
                if process.stderr:
                    print(f"--- Standard Error ---\n{process.stderr}")

        return process.returncode
    except Exception as e:
        if console:
            console.print(f"\n[bold red]✗ Error executing command:[/bold red] {e}")
        else:
            print(f"\nERROR: Error executing command: {e}")
        return 1


def check_python_version_simple():
    """Simple python version check using standard prints."""
    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10 or higher is required.")
        sys.exit(1)
    print(f"OK: Python {platform.python_version()} detected.")


def require_virtual_environment():
    """Checks whether a virtual environment is active.
    If a virtual environment is not active, it stops the installation process."""
    is_venv = sys.prefix != sys.base_prefix
    is_conda = "CONDA_PREFIX" in os.environ

    if is_venv or is_conda:
        return

    print("=" * 60)
    print("  ERROR: No virtual environment detected!")
    print("=" * 60)
    print()
    print("  This project installs PyTorch and other heavy")
    print("  dependencies. A virtual environment is required.")
    print()
    print("  Option A (venv):")
    print("    python -m venv .venv")
    print("    source .venv/bin/activate")
    print()
    print("  Option B (Conda):")
    print("    conda create -n arelto python=3.10")
    print("    conda activate arelto")
    print()
    print("  Then re-run:  python scripts/install_arelto.py")
    print("=" * 60)
    sys.exit(1)


def bootstrap_rich():
    """Bootstraps the 'rich' library silently and returns the UI components."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.prompt import Confirm

        return Console(), Panel, Confirm
    except ImportError:
        print("--> Installing 'rich' for a nicer experience...")
        try:
            process = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "rich"],
                capture_output=True,
                text=True,
            )
            if process.returncode != 0:
                print(f"\nERROR: Failed to install 'rich'.")
                print(f"Pip Error Output:\n{process.stderr}")
                print("Please install it manually: pip install rich")
                sys.exit(1)

            from rich.console import Console
            from rich.panel import Panel
            from rich.prompt import Confirm

            return Console(), Panel, Confirm
        except Exception as e:
            print(f"\nERROR: Failed to execute pip to install 'rich'.")
            print(f"Technical error: {e}")
            sys.exit(1)


def show_environment_info(console):
    """Display the active environment using Rich."""
    if sys.prefix != sys.base_prefix:
        env_name = os.path.basename(sys.prefix)
        console.print(
            f"[bold green]\u2713[/bold green] Virtual environment: [cyan]{env_name}[/cyan]"
        )
    elif "CONDA_PREFIX" in os.environ:
        env_name = os.path.basename(os.environ["CONDA_PREFIX"])
        console.print(
            f"[bold green]\u2713[/bold green] Conda environment: [cyan]{env_name}[/cyan]"
        )


def check_python_version(console):
    """Version check using rich."""
    console.print(
        f"[bold green]✓[/bold green] Python {platform.python_version()} detected."
    )


def check_git_submodules(console):
    if not os.path.exists(".git"):
        console.print(
            "[bold yellow]⚠[/bold yellow] Not a git repository. "
            "Skipping submodule update."
        )
        return

    submodules_populated = True
    if os.path.exists("extern"):
        if not os.listdir("extern"):
            submodules_populated = False
    else:
        submodules_populated = False

    if not submodules_populated:
        with console.status(
            "[bold cyan]Initializing git submodules...", spinner="dots"
        ):
            cmd = ["git", "submodule", "update", "--init", "--recursive"]
            ret = run_command(cmd, console=console)
            if ret != 0:
                console.print(
                    "[bold red]✗ ERROR: Failed to update submodules.[/bold red]"
                )
                sys.exit(1)
            console.print("[bold green]✓[/bold green] Submodules initialized.")
    else:
        console.print("[bold green]✓[/bold green] Submodules appear to be initialized.")


def check_system_dependencies(console):
    system = platform.system()
    missing = []

    for tool in ["cmake", "ninja"]:
        if shutil.which(tool) is None:
            missing.append(tool)

    if system == "Linux":
        if shutil.which("sdl2-config") is None:
            msg = (
                "[bold yellow]⚠[/bold yellow] 'sdl2-config' not found. "
                "Ensure SDL2 development libraries are installed.\n"
                "Suggested commands:\n"
                "  [cyan]Ubuntu/Debian:[/cyan] sudo apt install "
                "libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-mixer-dev\n"
                "  [cyan]Arch:[/cyan] sudo pacman -S sdl2 sdl2_image sdl2_ttf sdl2_mixer"
            )
            console.print(msg)

    if missing:
        tools_str = ", ".join(missing)
        console.print(f"[bold red]✗ Build tools are missing:[/bold red] {tools_str}")
        console.print("Please install them using your system package manager.")
        sys.exit(1)
    else:
        console.print("[bold green]✓[/bold green] Build tools (cmake, ninja) found.")


def clean_old_artifacts(console):
    rl_dir = "rl"
    cleaned = False

    extensions_to_clean = (".so", ".pyi")

    if os.path.exists(rl_dir):
        for item in os.listdir(rl_dir):
            if item.endswith(extensions_to_clean):
                os.remove(os.path.join(rl_dir, item))
                console.print(f"  [dim]- Removed old artifact: {item}[/dim]")
                cleaned = True

    if cleaned:
        console.print("[bold green]✓[/bold green] Cleanup finished.")
    else:
        console.print("[bold green]✓[/bold green] Workspace clean.")


def install_package(console):
    with console.status("[bold cyan]Configuring CMake...", spinner="bouncingBar"):
        cmake_cmd = ["cmake", "-B", "build", "-G", "Ninja", "-DCMAKE_INSTALL_PREFIX=."]
        ret = run_command(cmake_cmd, console=console)
        if ret != 0:
            console.print("[bold red]✗ ERROR: CMake configuration failed.[/bold red]")
            sys.exit(1)
        console.print("[bold green]✓[/bold green] CMake configured successfully.")

    with console.status(
        "[bold cyan]Building with Ninja (this may take a while)...",
        spinner="bouncingBar",
    ):
        ret = run_command(["ninja", "-C", "build", "install"], console=console)
        if ret != 0:
            console.print("[bold red]✗ ERROR: Ninja build/install failed.[/bold red]")
            sys.exit(1)
        console.print("[bold green]✓[/bold green] Build completed.")

    with console.status(
        "[bold cyan]Installing Python package (editable)...", spinner="bouncingBar"
    ):
        pip_cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
        ret = run_command(pip_cmd, console=console)
        if ret != 0:
            console.print(
                "\n[bold yellow]⚠ WARNING: Pip editable installation failed, but build succeeded.[/bold yellow]"
            )
        else:
            console.print("[bold green]✓[/bold green] Arelto installed successfully.")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    os.chdir(root_dir)

    # A correct python version and an activate venv are hard requirements.
    # Checking these first to ensure that the Rich formatted installation
    # procedure will be displayed as intended.
    check_python_version_simple()
    require_virtual_environment()

    console, Panel, Confirm = bootstrap_rich()

    console.print(
        Panel(
            "[bold white]Arelto Setup & Installation[/bold white]",
            style="blue",
            expand=False,
        )
    )
    console.print()

    # Pass console explicitly where needed
    show_environment_info(console)
    check_system_dependencies(console)
    check_git_submodules(console)
    clean_old_artifacts(console)

    console.print()
    install_package(console)

    console.print()
    console.print(
        Panel(
            "[bold green]Setup finished![/bold green]\n\n"
            "You can now run the game using:\n"
            "[cyan]python scripts/start_game_async.py[/cyan]",
            border_style="green",
            expand=False,
        )
    )


if __name__ == "__main__":
    main()
