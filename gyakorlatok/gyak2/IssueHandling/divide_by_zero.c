#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    const char* kernel_src =
        "__kernel void div_zero(__global float* A, __global float* B, __global float* C){\n"
        "   int id = get_global_id(0);\n"
        "   C[id] = A[id] / B[id];\n"
        "}";

    float A[4] = {1,2,3,4};
    float B[4] = {1,0,3,4}; // B[1] = 0 → hibás
    float C[4];

    cl_platform_id platform; cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(A), A, NULL);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(B), B, NULL);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(C), NULL, NULL);

    cl_program program = clCreateProgramWithSource(context, 1, &kernel_src, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "div_zero", NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);

    size_t global = 4;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    clFinish(queue);

    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(C), C, 0, NULL, NULL);

    printf("Eredmeny:\n");
    for (int i=0;i<4;i++) printf("%f\n", C[i]);

    clReleaseMemObject(bufA); clReleaseMemObject(bufB); clReleaseMemObject(bufC);
    clReleaseKernel(kernel); clReleaseProgram(program);
    clReleaseCommandQueue(queue); clReleaseContext(context);

    return 0;
}