import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from rl.algorithms.async_ppo import TrainingMetrics
from rl.utils.live_metrics_plotter import LiveMetricsPlotter


@pytest.fixture(autouse=True)
def _disable_figure_show(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)


def _metrics(samples: int, offset: float = 0.0) -> TrainingMetrics:
    return TrainingMetrics(
        trained_samples=samples,
        policy_loss=1.0 + offset,
        value_loss=2.0 + offset,
        mean_total_reward=3.0 + offset,
    )


def test_plotter_appends_full_history_and_resets() -> None:
    plotter = LiveMetricsPlotter(allow_noninteractive=True)
    try:
        plotter.add_metrics(_metrics(100))
        plotter.add_metrics(_metrics(200, 1.0))

        assert len(plotter.axes) == 3
        assert plotter.trained_samples == [100, 200]
        assert list(plotter.lines[0].get_xdata()) == [100, 200]
        assert list(plotter.lines[0].get_ydata()) == [1.0, 2.0]
        assert list(plotter.lines[1].get_ydata()) == [2.0, 3.0]
        assert list(plotter.lines[2].get_ydata()) == [3.0, 4.0]

        plotter.reset()

        assert plotter.trained_samples == []
        assert all(len(line.get_xdata()) == 0 for line in plotter.lines)
        assert all(len(line.get_ydata()) == 0 for line in plotter.lines)
    finally:
        plotter.close()


def test_closed_plotter_does_not_reopen_or_accept_metrics() -> None:
    plotter = LiveMetricsPlotter(allow_noninteractive=True)
    figure_number = plotter.figure.number

    plotter.close()
    plotter.add_metrics(_metrics(100))
    plotter.process_events()

    assert not plotter.active
    assert plotter.trained_samples == []
    assert not plotter._is_open()
    assert figure_number not in plt.get_fignums()
