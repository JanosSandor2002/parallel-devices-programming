__kernel void hello(__global float* data)
{
    int id = get_global_id(0);
    data[id] = data[id] * 2.0f;
}