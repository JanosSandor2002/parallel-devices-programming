#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    const char* kernel_src =
        "__kernel void invalid(__global float* A){\n"
        "   float* tmp = (float*)malloc(sizeof(float)*4);\n" // HIBA!
        "   tmp[0] = 1.0f;\n"
        "}";

    cl_platform_id platform; cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_src, NULL, NULL);

    cl_int err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Build hiba:\n%s\n", log);
        free(log);
    }

    clReleaseProgram(program);
    clReleaseContext(context);
    return 0;
}