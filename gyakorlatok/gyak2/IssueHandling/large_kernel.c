#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    /* Nagy kernel string generálása (például 10 MB) */
    size_t N = 10 * 1024 * 1024;
    char* kernel_source = (char*)malloc(N);
    if (!kernel_source) return 1;

    for (size_t i = 0; i < N-1; i++) kernel_source[i] = ' ';
    kernel_source[N-1] = '\0';

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &N, NULL);
    cl_int err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    if (err != CL_SUCCESS) {
        printf("Hiba a kernel build során: %d\n", err);
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Build log:\n%s\n", log);
        free(log);
    }

    free(kernel_source);
    clReleaseProgram(program);
    clReleaseContext(context);

    return 0;
}