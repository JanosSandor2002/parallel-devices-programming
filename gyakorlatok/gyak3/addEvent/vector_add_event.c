#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define N 8

const char* kernelSource =
"__kernel void add(__global float* A, __global float* B, __global float* C){"
"   int id = get_global_id(0);"
"   C[id] = A[id] + B[id];"
"}";

/* ===== Kernel callback ===== */
void CL_CALLBACK kernel_finished(
    cl_event event,
    cl_int status,
    void* user_data)
{
    printf("Kernel vegrehajtas befejezodott!\n");
}

/* ===== Read buffer callback ===== */
void CL_CALLBACK read_finished(
    cl_event event,
    cl_int status,
    void* user_data)
{
    float* result = (float*)user_data;

    printf("Buffer visszaolvasva:\n");
    for(int i=0;i<N;i++)
        printf("%f ", result[i]);
    printf("\n");
}

int main() {

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    clGetPlatformIDs(1,&platform,NULL);
    clGetDeviceIDs(platform,CL_DEVICE_TYPE_DEFAULT,1,&device,NULL);

    context = clCreateContext(NULL,1,&device,NULL,NULL,NULL);

    queue = clCreateCommandQueue(
        context,
        device,
        CL_QUEUE_PROFILING_ENABLE,
        NULL);

    /* ===== program ===== */
    cl_program program =
        clCreateProgramWithSource(context,1,&kernelSource,NULL,NULL);

    clBuildProgram(program,0,NULL,NULL,NULL,NULL);

    cl_kernel kernel = clCreateKernel(program,"add",NULL);

    /* ===== adatok ===== */
    float A[N], B[N], C[N];

    for(int i=0;i<N;i++){
        A[i]=i;
        B[i]=i*2;
    }

    cl_mem bufA = clCreateBuffer(context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float)*N, A, NULL);

    cl_mem bufB = clCreateBuffer(context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float)*N, B, NULL);

    cl_mem bufC = clCreateBuffer(context,
        CL_MEM_WRITE_ONLY,
        sizeof(float)*N, NULL, NULL);

    clSetKernelArg(kernel,0,sizeof(cl_mem),&bufA);
    clSetKernelArg(kernel,1,sizeof(cl_mem),&bufB);
    clSetKernelArg(kernel,2,sizeof(cl_mem),&bufC);

    size_t global=N;

    /* ===== kernel esemény ===== */
    cl_event kernel_event;

    clEnqueueNDRangeKernel(
        queue,
        kernel,
        1,
        NULL,
        &global,
        NULL,
        0,
        NULL,
        &kernel_event);

    /* callback kernel végére */
    clSetEventCallback(
        kernel_event,
        CL_COMPLETE,
        kernel_finished,
        NULL);

    /* ===== buffer read esemény ===== */
    cl_event read_event;

    clEnqueueReadBuffer(
        queue,
        bufC,
        CL_FALSE,   // ASYNC !!!
        0,
        sizeof(float)*N,
        C,
        1,
        &kernel_event, // kernel után indul
        &read_event);

    /* callback a read végére */
    clSetEventCallback(
        read_event,
        CL_COMPLETE,
        read_finished,
        C);

    /* várunk mindenre */
    clFinish(queue);

    /* cleanup */
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}