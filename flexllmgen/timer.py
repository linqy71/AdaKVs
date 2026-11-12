"""Global timer for profiling."""
from collections import namedtuple
import time
from typing import Callable, Any


class _Timer:
    """An internal timer."""

    def __init__(self, name: str):
        self.name = name
        self.started = False
        self.start_time = None

        # start-stop timestamp pairs
        self.start_times = []
        self.stop_times = []
        self._costs = []
        self._cuda_pending = []
        self._cuda_start = None

    @property
    def costs(self):
        # Resolve only when results are consumed, normally after request sync.
        for start, end in self._cuda_pending:
            end.synchronize()
            self._costs.append(start.elapsed_time(end) / 1000.0)
        self._cuda_pending.clear()
        return self._costs

    def start_cuda(self, stream=None):
        import torch
        assert not self.started
        self._cuda_start = torch.cuda.Event(enable_timing=True)
        self._cuda_start.record(stream)
        self.started = True

    def stop_cuda(self, stream=None):
        import torch
        assert self.started and self._cuda_start is not None
        end = torch.cuda.Event(enable_timing=True)
        end.record(stream)
        self._cuda_pending.append((self._cuda_start, end))
        self._cuda_start = None
        self.started = False

    def start(self, sync_func: Callable = None):
        """Start the timer."""
        assert not self.started, f"timer {self.name} has already been started."
        if sync_func:
            sync_func()

        self.start_time = time.perf_counter()
        self.start_times.append(self.start_time)
        self.started = True

    def stop(self, sync_func: Callable = None):
        """Stop the timer."""
        assert self.started, f"timer {self.name} is not started."
        if sync_func:
            sync_func()

        stop_time = time.perf_counter()
        self.costs.append(stop_time - self.start_time)
        self.stop_times.append(stop_time)
        self.started = False

    def reset(self):
        """Reset timer."""
        self.started = False
        self.start_time = None
        self.start_times = []
        self.stop_times = []
        self._costs = []
        self._cuda_pending = []
        self._cuda_start = None

    def elapsed(self, mode: str = "average"):
        """Calculate the elapsed time."""
        if not self.costs:
            return 0.0
        if mode == "average":
            return sum(self.costs) / len(self.costs)
        elif mode == "sum":
            return sum(self.costs)
        else:
            raise RuntimeError("Supported mode is: average | sum")


class Timers:
    """A group of timers."""

    def __init__(self):
        self.timers = {}

    def __call__(self, name: str):
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def __contains__(self, name: str):
        return name in self.timers


timers = Timers()

Event = namedtuple("Event", ("tstamp", "name", "info"))


class Tracer:
    """An activity tracer."""

    def __init__(self):
        self.events = []

    def log(self, name: str, info: Any, sync_func: Callable = None):
        if sync_func:
            sync_func()

        self.events.append(Event(time.perf_counter(), name, info))


tracer = Tracer()
