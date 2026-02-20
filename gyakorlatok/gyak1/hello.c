#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

const char* kernel_code =
    "__kernel void hello_kernel(__global int* buffer, int n) {\n"
    "   int gid = get_global_id(0);\n"
    "   if (gid < n) {\n"
    "       buffer[gid] = 11;\n"
    "   }\n"
    "}\n";

const int SAMPLE_SIZE = 1000;

int main(void)
{
    cl_int err;

    // 1️⃣ Get platform
    cl_uint n_platforms;
    cl_platform_id platform_id;
    err = clGetPlatformIDs(1, &platform_id, &n_platforms);
    if (err != CL_SUCCESS || n_platforms == 0) {
        printf("[ERROR] clGetPlatformIDs failed. Code: %d\n", err);
        return 1;
    }

    // 2️⃣ Get device
    cl_device_id device_id;
    cl_uint n_devices;
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &n_devices);
    if (err != CL_SUCCESS || n_devices == 0) {
        printf("[ERROR] clGetDeviceIDs failed. Code: %d\n", err);
        return 1;
    }

    // 3️⃣ Create OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clCreateContext failed. Code: %d\n", err);
        return 1;
    }

    // 4️⃣ Build the program
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_code, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clCreateProgramWithSource failed. Code: %d\n", err);
        clReleaseContext(context);
        return 1;
    }

    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Get build log
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("[ERROR] clBuildProgram failed. Log:\n%s\n", log);
        free(log);
        clReleaseProgram(program);
        clReleaseContext(context);
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "hello_kernel", &err);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clCreateKernel failed. Code: %d\n", err);
        clReleaseProgram(program);
        clReleaseContext(context);
        return 1;
    }

    // 5️⃣ Create host buffer
    int* host_buffer = (int*)malloc(SAMPLE_SIZE * sizeof(int));
    for (int i = 0; i < SAMPLE_SIZE; ++i) host_buffer[i] = i;

    // 6️⃣ Create device buffer
    cl_mem device_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, SAMPLE_SIZE * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clCreateBuffer failed. Code: %d\n", err);
        free(host_buffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseContext(context);
        return 1;
    }

    // 7️⃣ Set kernel arguments
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(int), &SAMPLE_SIZE);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clSetKernelArg failed. Code: %d\n", err);
        clReleaseMemObject(device_buffer);
        free(host_buffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseContext(context);
        return 1;
    }

    // 8️⃣ Create modern command queue
    cl_queue_properties props[] = {0};
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, props, &err);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clCreateCommandQueueWithProperties failed. Code: %d\n", err);
        clReleaseMemObject(device_buffer);
        free(host_buffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseContext(context);
        return 1;
    }

    // 9️⃣ Write host buffer -> device buffer
    err = clEnqueueWriteBuffer(command_queue, device_buffer, CL_TRUE, 0, SAMPLE_SIZE * sizeof(int), host_buffer, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clEnqueueWriteBuffer failed. Code: %d\n", err);
    }

    // 10️⃣ Launch kernel
    size_t local_work_size = 256;
    size_t n_work_groups = (SAMPLE_SIZE + local_work_size - 1) / local_work_size;
    size_t global_work_size = n_work_groups * local_work_size;

    err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clEnqueueNDRangeKernel failed. Code: %d\n", err);
    }

    // 11️⃣ Read device buffer -> host buffer
    err = clEnqueueReadBuffer(command_queue, device_buffer, CL_TRUE, 0, SAMPLE_SIZE * sizeof(int), host_buffer, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clEnqueueReadBuffer failed. Code: %d\n", err);
    }

    // 12️⃣ Print results
    for (int i = 0; i < SAMPLE_SIZE; ++i) {
        printf("[%d] = %d\n", i, host_buffer[i]);
    }

    // 13️⃣ Cleanup
    clReleaseCommandQueue(command_queue);
    clReleaseMemObject(device_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    free(host_buffer);

    return 0;
}
