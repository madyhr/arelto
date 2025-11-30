import time
import collections
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Group
from rich import box

# --- Safety Imports ---
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pynvml
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

class GameDiagnostics:
    def __init__(self, target_fps=60.0, history_len=60):
        self.target_fps = target_fps
        self.target_ms = 1000.0 / target_fps
        self.stats = collections.defaultdict(lambda: collections.deque(maxlen=history_len))
        
        self.process = psutil.Process() if HAS_PSUTIL else None
        
        self.gpu_handle = None
        if HAS_GPU:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except:
                self.gpu_handle = None

    def __enter__(self):
        self.live = Live(self.generate_layout(), refresh_per_second=4)
        self.live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.live.stop()
        if HAS_GPU and self.gpu_handle:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

    def record(self, category: str, value: float):
        if category == 'fps':
            # FPS is already a number/frequency, store as is
            self.stats[category].append(value)
        else:
            # Other categories are time durations (seconds), convert to ms
            self.stats[category].append(value * 1000.0)    

    def tick(self):
        self.live.update(self.generate_layout())

    def _make_bar(self, value, total, color="green", critical_threshold=0.9):
        if total == 0: total = 1
        percent = min(1.0, value / total)
        blocks = int(percent * 20)
        if percent > critical_threshold: color = "red"
        bar_str = f"[{color}]{'█' * blocks}{'░' * (20 - blocks)}[/]"
        return f"{bar_str} {value:.1f}/{total:.1f}"

    def generate_layout(self):
        timing_table = Table(box=box.SIMPLE, expand=True)
        timing_table.add_column("System", style="cyan")
        timing_table.add_column("Avg (ms)", justify="right")
        timing_table.add_column("Budget", justify="left")

        for category, times in self.stats.items():
            if category == 'fps': continue
            avg = sum(times) / len(times) if times else 0
            timing_table.add_row(category.capitalize(), f"{avg:.3f}", self._make_bar(avg, self.target_ms, critical_threshold=1.0))

        fps_hist = self.stats['fps']
        avg_fps = sum(fps_hist) / len(fps_hist) if fps_hist else 0
        fps_panel = Panel(f"[bold {'green' if avg_fps > self.target_fps * 0.9 else 'red'}]FPS: {avg_fps:.1f}[/]", title="Performance", border_style="white")

        hw_table = Table(box=box.SIMPLE, expand=True, show_header=False)
        if HAS_PSUTIL:
            cpu_pct = psutil.cpu_percent(interval=None)
            mem_info = self.process.memory_info()
            ram_mb = mem_info.rss / 1024 / 1024
            hw_table.add_row("CPU", self._make_bar(cpu_pct, 100, "blue"))
            hw_table.add_row("RAM (MB)", f"[blue]{ram_mb:.1f} MB[/]")
        
        if HAS_GPU and self.gpu_handle:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                hw_table.add_row("GPU Core", self._make_bar(util.gpu, 100, "magenta"))
                hw_table.add_row("VRAM", self._make_bar(mem.used / 1024 / 1024, mem.total / 1024 / 1024, "magenta"))
            except:
                hw_table.add_row("GPU", "[yellow]Error[/]")

        layout = Layout()
        layout.split_row(Layout(Group(fps_panel, timing_table), name="left"), Layout(Panel(hw_table, title="Hardware"), name="right"))
        return layout
