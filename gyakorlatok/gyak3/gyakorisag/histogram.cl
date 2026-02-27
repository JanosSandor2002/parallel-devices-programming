__kernel void histogram(
    __global const int* data,
    __global int* freq)
{
    int id = get_global_id(0);

    int value = data[id];

    atomic_add(&freq[value], 1);
}