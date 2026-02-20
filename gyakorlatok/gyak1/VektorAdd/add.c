#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define N 8

/* kernel fajl beolvasasa */
char* readKernel(const char* filename, size_t* size_out) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        printf("Nem sikerult megnyitni a kernel fajlt\n");
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);

    char* src = (char*)malloc(size + 1);
    fread(src, 1, size, f);
    src[size] = '\0';

    fclose(f);

    *size_out = size;
    return src;
}

int main() {

    /* host adatok */
    float A[N], B[N], C[N];

    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i * 10;
    }

    /* platform + device */
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);

    /* context */
    cl_context context =
        clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    /* command queue */
    cl_command_queue queue =
        clCreateCommandQueue(context, device, 0, NULL);

    /* bufferok */
    cl_mem bufA = clCreateBuffer(context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float)*N, A, NULL);

    cl_mem bufB = clCreateBuffer(context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float)*N, B, NULL);

    cl_mem bufC = clCreateBuffer(context,
        CL_MEM_WRITE_ONLY,
        sizeof(float)*N, NULL, NULL);

    /* kernel betoltese */
    size_t src_size;
    char* source = readKernel("vector_add.cl", &src_size);

    cl_program program =
        clCreateProgramWithSource(context, 1,
                                  (const char**)&source,
                                  &src_size, NULL);

    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    cl_kernel kernel =
        clCreateKernel(program, "vector_add", NULL);

    /* kernel argumentumok */
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);

    /* futtatas */
    size_t global = N;

    clEnqueueNDRangeKernel(queue,
                           kernel,
                           1,
                           NULL,
                           &global,
                           NULL,
                           0, NULL, NULL);

    /* eredmeny vissza */
    clEnqueueReadBuffer(queue, bufC,
                        CL_TRUE,
                        0,
                        sizeof(float)*N,
                        C,
                        0, NULL, NULL);

    /* kiiras */
    printf("Eredmeny:\n");
    for (int i = 0; i < N; i++)
        printf("%f + %f = %f\n", A[i], B[i], C[i]);

    /* takaritas */
    free(source);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}