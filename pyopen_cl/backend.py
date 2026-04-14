# opencl backend
import math
import numpy as np
import pyopencl as cl


def get_device():
    platforms = cl.get_platforms()
    for plat in platforms:
        gpus = plat.get_devices(device_type=cl.device_type.GPU)
        if gpus:
            return gpus[0]
    return platforms[0].get_devices()[0]


def build_context(device):
    ctx = cl.Context(devices=[device])
    queue = cl.CommandQueue(
        ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE
    )
    return ctx, queue


def load_kernel(ctx, kernel_path):
    with open(kernel_path, "r") as f:
        source = f.read()
    program = cl.Program(ctx, source).build()

    knl_sum = cl.Kernel(program, "sum_of_squares")
    knl_sum.set_scalar_arg_dtypes([None, None, np.int32])

    knl_reduce = cl.Kernel(program, "reduce")
    knl_reduce.set_scalar_arg_dtypes([None, None, np.int32])

    return knl_sum, knl_reduce


def run_kernel(queue, kernels, n, input_buf):
    knl_sum, knl_reduce = kernels
    mf = cl.mem_flags
    device = queue.device
    local_size = min(256, device.max_work_group_size)

    events = []

    # ── 1. pass: x[i]^2, minden work-group összegzi a saját blokkját ──
    num_groups = math.ceil(n / local_size)
    partial_buf = cl.Buffer(
        queue.context, mf.READ_WRITE,
        size=num_groups * np.dtype(np.float32).itemsize
    )
    ev = knl_sum(
        queue,
        (num_groups * local_size,),
        (local_size,),
        input_buf,
        partial_buf,
        np.int32(n)
    )
    events.append(ev)

    # ── További pass-ok: partial -> partial, amíg 1 elem nem marad ────
    current_buf = partial_buf
    current_n = num_groups

    while current_n > 1:
        next_groups = math.ceil(current_n / local_size)
        next_buf = cl.Buffer(
            queue.context, mf.READ_WRITE,
            size=next_groups * np.dtype(np.float32).itemsize
        )
        ev = knl_reduce(
            queue,
            (next_groups * local_size,),
            (local_size,),
            current_buf,
            next_buf,
            np.int32(current_n)
        )
        events.append(ev)
        current_buf = next_buf
        current_n = next_groups

    return current_buf, events


def next_power_of_two(x: int) -> int:
    return 1 if x == 0 else 2 ** math.ceil(math.log2(x))