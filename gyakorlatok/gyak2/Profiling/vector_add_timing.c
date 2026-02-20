#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1000000

/* kernel file beolvasása */
char* readKernel(const char* filename, size_t* size_out) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;
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

/* host time measurement */
double diff_ms(clock_t start, clock_t end) {
    return ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0; // ms
}

int main() {
    clock_t t0, t1;
    double times[4]; // kernel_load, buffer, run, read

    float *A = malloc(sizeof(float)*N);
    float *B = malloc(sizeof(float)*N);
    float *C = malloc(sizeof(float)*N);
    for (int i=0;i<N;i++){ A[i]=i; B[i]=i*2; }

    /* 1. Kernel load + build */
    t0 = clock();
    size_t src_size;
    char* source = readKernel("vector_add.cl",&src_size);

    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1,&platform,NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);
    cl_context context = clCreateContext(NULL,1,&device,NULL,NULL,NULL);
    cl_program program = clCreateProgramWithSource(context,1,(const char**)&source,&src_size,NULL);
    clBuildProgram(program,1,&device,NULL,NULL,NULL);
    cl_kernel kernel = clCreateKernel(program,"vector_add",NULL);
    t1 = clock(); times[0] = diff_ms(t0,t1);

    /* 2. Buffer készítés */
    t0 = clock();
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*N, A,NULL);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*N, B,NULL);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*N, NULL,NULL);
    t1 = clock(); times[1] = diff_ms(t0,t1);

    /* 3. Kernel futtatása */
    t0 = clock();
    clSetKernelArg(kernel,0,sizeof(cl_mem),&bufA);
    clSetKernelArg(kernel,1,sizeof(cl_mem),&bufB);
    clSetKernelArg(kernel,2,sizeof(cl_mem),&bufC);

    size_t global = N;
    cl_command_queue queue = clCreateCommandQueue(context,device,0,NULL);
    clEnqueueNDRangeKernel(queue,kernel,1,NULL,&global,NULL,0,NULL,NULL);
    clFinish(queue);
    t1 = clock(); times[2] = diff_ms(t0,t1);

    /* 4. Eredmény visszaolvasása */
    t0 = clock();
    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(float)*N, C, 0,NULL,NULL);
    t1 = clock(); times[3] = diff_ms(t0,t1);

    printf("Times (ms): Kernel load/build=%.2f, Buffer=%.2f, Run=%.2f, Read=%.2f\n",
           times[0],times[1],times[2],times[3]);

    /* cleanup */
    free(A); free(B); free(C); free(source);
    clReleaseMemObject(bufA); clReleaseMemObject(bufB); clReleaseMemObject(bufC);
    clReleaseKernel(kernel); clReleaseProgram(program);
    clReleaseCommandQueue(queue); clReleaseContext(context);

    return 0;
}