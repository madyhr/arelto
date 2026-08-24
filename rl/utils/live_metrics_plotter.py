from __future__ import annotations

import matplotlib.pyplot as plt

from rl.algorithms.async_ppo import TrainingMetrics


class LiveMetricsPlotter:
    """Non-blocking live plots for asynchronous PPO training metrics."""

    def __init__(self, *, allow_noninteractive: bool = False) -> None:
        if not allow_noninteractive and plt.get_backend().lower() == "agg":
            try:
                plt.switch_backend("qtagg")
            except (ImportError, RuntimeError) as error:
                raise RuntimeError(
                    "Live metrics require the Matplotlib Qt backend. "
                    "Install the project dependencies to provide PyQt6."
                ) from error

        plt.ion()
        self.figure, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 10))
        self.axes = list(axes)
        self.figure.canvas.manager.set_window_title("Arelto Live Training Metrics")
        self.figure.canvas.mpl_connect("close_event", self._on_close)

        titles = ("Policy Loss", "Value Loss", "Mean Total Reward")
        y_labels = ("Policy loss", "Value loss", "Mean total reward")
        self.lines = []
        for axis, title, y_label in zip(self.axes, titles, y_labels):
            (line,) = axis.plot([], [])
            axis.set_title(title)
            axis.set_ylabel(y_label)
            axis.grid(True, alpha=0.3)
            self.lines.append(line)
        self.axes[-1].set_xlabel("Trained samples")

        self.trained_samples: list[int] = []
        self.policy_losses: list[float] = []
        self.value_losses: list[float] = []
        self.mean_total_rewards: list[float] = []
        self.active = True

        self.figure.tight_layout()
        plt.show(block=False)

    def add_metrics(self, metrics: TrainingMetrics) -> None:
        if not self._is_open():
            return

        self.trained_samples.append(metrics.trained_samples)
        self.policy_losses.append(metrics.policy_loss)
        self.value_losses.append(metrics.value_loss)
        self.mean_total_rewards.append(metrics.mean_total_reward)

        histories = (
            self.policy_losses,
            self.value_losses,
            self.mean_total_rewards,
        )
        for axis, line, history in zip(self.axes, self.lines, histories):
            line.set_data(self.trained_samples, history)
            axis.relim()
            axis.autoscale_view()

        self.figure.canvas.draw_idle()

    def process_events(self) -> None:
        if not self._is_open():
            return
        self.figure.canvas.flush_events()

    def reset(self) -> None:
        if not self._is_open():
            return

        self.trained_samples.clear()
        self.policy_losses.clear()
        self.value_losses.clear()
        self.mean_total_rewards.clear()

        for axis, line in zip(self.axes, self.lines):
            line.set_data([], [])
            axis.set_ylim(0.0, 1.0)
        self.axes[-1].set_xlim(0.0, 1.0)
        for axis in self.axes:
            axis.set_autoscalex_on(True)
            axis.set_autoscaley_on(True)
        self.figure.canvas.draw_idle()

    def close(self) -> None:
        if not self.active:
            return
        self.active = False
        plt.close(self.figure)

    def _is_open(self) -> bool:
        if self.active and not plt.fignum_exists(self.figure.number):
            self.active = False
        return self.active

    def _on_close(self, _event) -> None:
        self.active = False
