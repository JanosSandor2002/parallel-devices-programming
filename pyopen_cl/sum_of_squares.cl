__kernel void sum_of_squares(__global const float* input, __global float* output) {
    int gid = get_global_id(0);
    output[gid] = input[gid] * input[gid];
}