"""Two reusable head-major KV staging slots for single-request prefill."""
import numpy as np
import torch

from flexllmgen.timer import timers


class PrefixKVWorkspace:
    def __init__(self, heads, head_dim, device, host_copy_backend="torch"):
        if host_copy_backend not in ("torch", "numpy"):
            raise ValueError("Unknown KV staging host copy backend")
        self.host_copy_backend = host_copy_backend
        self.heads, self.head_dim, self.device = heads, head_dim, device
        self.stream = torch.cuda.Stream(device=device)
        self.slots = [None, None]
        self.next_slot = 0
        self.direct_slots = [None, None]
        self.next_direct_slot = 0

    def acquire_direct(self, max_prefix_len, suffix_len):
        """Acquire host slabs before the collector writes, without a CPU copy.

        The planner chooses the compact head stride later. Reserve worst-case
        host storage once, but DMA and GPU storage use only the live width.
        """
        index = self.next_direct_slot
        self.next_direct_slot = (index + 1) % 2
        slot = self.direct_slots[index]
        needed = self.heads * (max_prefix_len + suffix_len) * self.head_dim
        if slot is None:
            slot = dict(host_capacity=0, gpu_capacity=0, ready=None, consumed=None)
            self.direct_slots[index] = slot
        if slot['ready'] is not None:
            slot['ready'].synchronize()  # Prior DMA must finish before native scatter.
        if slot['host_capacity'] < needed:
            slot['host_k'] = torch.empty(needed, dtype=torch.float16, pin_memory=True)
            slot['host_v'] = torch.empty(needed, dtype=torch.float16, pin_memory=True)
            slot['host_capacity'] = needed
        return slot

    def transfer_direct(self, slot, prefix_len, suffix_len):
        """Transfer compact native output, never the slab's unused capacity."""
        width = prefix_len + suffix_len
        needed = self.heads * width * self.head_dim
        if needed > slot['host_capacity']:
            raise ValueError('Direct collector exceeded the acquired slab')
        if slot['gpu_capacity'] < needed:
            if slot['consumed'] is not None:
                slot['consumed'].synchronize()
            slot['gpu_k'] = torch.empty(needed, dtype=torch.float16, device=self.device)
            slot['gpu_v'] = torch.empty(needed, dtype=torch.float16, device=self.device)
            slot['gpu_capacity'] = needed
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
        slot['key'] = slot['gpu_k'][:needed].view(self.heads, width, self.head_dim)
        slot['value'] = slot['gpu_v'][:needed].view(self.heads, width, self.head_dim)
        with torch.cuda.stream(self.stream):
            if slot['consumed'] is not None:
                self.stream.wait_event(slot['consumed'])
            timers('copy prefix').start_cuda(self.stream)
            slot['gpu_k'][:needed].copy_(slot['host_k'][:needed], non_blocking=True)
            slot['gpu_v'][:needed].copy_(slot['host_v'][:needed], non_blocking=True)
            timers('copy prefix').stop_cuda(self.stream)
            slot['ready'] = torch.cuda.Event()
            slot['ready'].record(self.stream)
        torch.cuda.current_stream(self.device).wait_event(slot['ready'])
        return slot

    def stage(self, key, value, prefix_len, suffix_len, valid_counts=None):
        index = self.next_slot
        self.next_slot = (index + 1) % 2
        slot = self.slots[index]
        capacity = prefix_len + suffix_len
        if slot is not None:
            # Host memory must not be overwritten while its DMA is in flight.
            slot["ready"].synchronize()
        if slot is None or slot["capacity"] < capacity:
            if slot is not None and slot["consumed"] is not None:
                slot["consumed"].synchronize()
            shape = (self.heads, capacity, self.head_dim)
            slot = {
                "capacity": capacity,
                "host_k": torch.empty(shape, dtype=torch.float16, pin_memory=True),
                "host_v": torch.empty(shape, dtype=torch.float16, pin_memory=True),
                "key": torch.empty(shape, dtype=torch.float16, device=self.device),
                "value": torch.empty(shape, dtype=torch.float16, device=self.device),
                "ready": torch.cuda.Event(), "consumed": None,
            }
            # GPU storage is allocated on the compute stream. The transfer
            # stream must respect any preceding reuse of that allocator block.
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            self.slots[index] = slot
        timers("kv postprocess").start()
        # Contiguous head rows, no CPU head/token transpose. Include a small
        # suffix reserve in DMA to keep the transfer contiguous for all heads.
        self.copy_host_prefix(slot["host_k"], key, prefix_len,
                              self.host_copy_backend, valid_counts)
        self.copy_host_prefix(slot["host_v"], value, prefix_len,
                              self.host_copy_backend, valid_counts)
        timers("kv postprocess").stop()
        with torch.cuda.stream(self.stream):
            if slot["consumed"] is not None:
                self.stream.wait_event(slot["consumed"])
            timers("copy prefix").start_cuda(self.stream)
            slot["key"].copy_(slot["host_k"], non_blocking=True)
            slot["value"].copy_(slot["host_v"], non_blocking=True)
            timers("copy prefix").stop_cuda(self.stream)
            slot["ready"].record(self.stream)
        torch.cuda.current_stream(self.device).wait_event(slot["ready"])
        return slot

    @staticmethod
    def copy_host_prefix(destination, source, prefix_len, backend, valid_counts=None):
        """Copy contiguous head rows without a CPU OpenMP launch for Quest.

        NumPy views share the existing native source and pinned destination;
        no intermediate allocation or pin_memory() copy is introduced.
        The caller waits for the slot's previous DMA before writing here.
        """
        if valid_counts is not None:
            # Native collection packs valid rows first in each head. Copy
            # those rows directly and clear only the invalid tail; never scan
            # the full K/V payload with a broadcast masked_fill kernel.
            target, data = destination.numpy(), source.numpy()
            for head, count in enumerate(valid_counts):
                np.copyto(target[head, :count], data[head, :count], casting="no")
                target[head, count:prefix_len].fill(0)
        elif backend == "numpy":
            np.copyto(destination[:, :prefix_len].numpy(),
                      source[:, :prefix_len].numpy(), casting="no")
        else:
            destination[:, :prefix_len].copy_(source[:, :prefix_len])

    @staticmethod
    def release(slot):
        # The attention kernels have been enqueued on the current stream.
        # Next use waits on this event rather than synchronizing the device.
        event = torch.cuda.Event()
        event.record()
        slot["consumed"] = event
