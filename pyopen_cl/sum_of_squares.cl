// 1. pass: négyzetösszeg
__kernel void sum_of_squares(__global const float* input,
                              __global float* output,
                              const int n)
{
    __local float scratch[256];
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    scratch[lid] = (gid < n) ? (input[gid] * input[gid]) : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) output[get_group_id(0)] = scratch[0];
}

// 2..k. pass: sima összeadás (már négyzetek vannak)
__kernel void reduce(__global const float* input,
                     __global float* output,
                     const int n)
{
    __local float scratch[256];
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    scratch[lid] = (gid < n) ? input[gid] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) output[get_group_id(0)] = scratch[0];
}