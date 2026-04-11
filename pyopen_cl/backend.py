# opencl backend
import pyopencl as cl
import numpy as np

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
    return cl.Program(ctx, source).build()


def run_kernel(queue, program, n, input_buf, output_buf):
    event = program.sum_of_squares(queue, (n,), None, input_buf, output_buf)
    return event