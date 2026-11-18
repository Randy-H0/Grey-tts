"""One batch ahead on a second thread (2026-11-07, you: "I don't want the cpu to be a bottleneck anymore").

Training is bound by the CPU launching GPU work from ONE thread. The VTV rows' targets (dacts/vtv.py ContentFeatures.batch:
codec decode + ContentVec + RMVPE per row, frozen, no gradient) were ~30% of that thread (py-spy 11-07). Here a worker
thread reads the loader and, for each batch, copies its latents to the GPU and builds those targets on its own CUDA stream,
while the main thread trains on the batch before. Torch releases the GIL while it dispatches, so the two threads' launches
overlap. The maths is unchanged: the same function on the same batch, one batch earlier.

  for batch, pre in Ahead(loader, make, device):   # pre = make(batch), or the exception make raised (re-raise it where
      ...                                           # the old inline call was, so an out-of-memory skip still works)
"""
import queue
import threading

import torch

_END, _FAIL = object(), object()


def _record(x, stream):
    """the caching allocator must not hand a tensor's memory to the worker's stream while the main stream still uses it."""
    if torch.is_tensor(x):
        if x.is_cuda:
            x.record_stream(stream)
    elif isinstance(x, (tuple, list)):
        for y in x:
            _record(y, stream)
    elif isinstance(x, dict):
        for y in x.values():
            _record(y, stream)


class Ahead:
    def __init__(self, loader, make, device, depth: int = 1):
        d = torch.device(device)
        if d.type == "cuda" and d.index is None:
            d = torch.device("cuda", torch.cuda.current_device())
        self.loader, self.make, self.dev, self.depth = loader, make, d, depth

    def __iter__(self):
        q = queue.Queue(maxsize=self.depth)
        stop = threading.Event()
        dev = self.dev

        def work():
            try:
                torch.cuda.set_device(dev)
                side = torch.cuda.Stream(device=dev)
                for b in self.loader:
                    if stop.is_set():
                        return
                    with torch.cuda.stream(side):
                        try:
                            r = self.make(b)
                        except Exception as e:  # noqa: BLE001  (handed to the main thread with its batch)
                            r = e
                        ev = torch.cuda.Event()
                        ev.record(side)
                    while not stop.is_set():
                        try:
                            q.put((b, r, ev), timeout=0.5)
                            break
                        except queue.Full:
                            pass
                q.put((_END, None, None))
            except BaseException as e:  # noqa: BLE001  (the loader itself failed: the main thread raises it)
                q.put((_FAIL, e, None))

        th = threading.Thread(target=work, name="ahead", daemon=True)
        th.start()
        try:
            while True:
                b, r, ev = q.get()
                if b is _END:
                    return
                if b is _FAIL:
                    raise r
                cur = torch.cuda.current_stream(dev)
                cur.wait_event(ev)  # (GPU-side wait: the main thread itself never blocks here)
                if not isinstance(r, BaseException):
                    _record(r, cur)
                yield b, r
        finally:
            stop.set()
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
