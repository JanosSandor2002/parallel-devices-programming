#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1000
#define MAX_VALUE 100

/* ===== kernel fájl beolvasása ===== */
char* loadKernel(const char* filename) {

    FILE* fp = fopen(filename, "r");
    if(!fp){
        printf("Nem sikerult megnyitni a kernel fajlt!\n");
        exit(1);
    }

    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    rewind(fp);

    char* source = (char*)malloc(size + 1);
    fread(source, 1, size, fp);
    source[size] = '\0';

    fclose(fp);
    return source;
}

int main() {

    srand(time(NULL));

    /* ===== adat generálás ===== */
    int data[N];
    for(int i=0;i<N;i++)
        data[i] = rand() % (MAX_VALUE + 1);

    int freq[MAX_VALUE+1] = {0};

    /* ===== OpenCL init ===== */
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    clGetPlatformIDs(1,&platform,NULL);
    clGetDeviceIDs(platform,CL_DEVICE_TYPE_DEFAULT,1,&device,NULL);

    context = clCreateContext(NULL,1,&device,NULL,NULL,NULL);
    queue = clCreateCommandQueue(context,device,0,NULL);

    /* ===== kernel betöltése ===== */
    char* source = loadKernel("histogram.cl");

    cl_program program =
        clCreateProgramWithSource(context,1,
            (const char**)&source,NULL,NULL);

    clBuildProgram(program,0,NULL,NULL,NULL,NULL);

    cl_kernel kernel =
        clCreateKernel(program,"histogram",NULL);

    /* ===== bufferek ===== */
    cl_mem dataBuf = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int)*N,
        data,
        NULL);

    cl_mem freqBuf = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(int)*(MAX_VALUE+1),
        freq,
        NULL);

    clSetKernelArg(kernel,0,sizeof(cl_mem),&dataBuf);
    clSetKernelArg(kernel,1,sizeof(cl_mem),&freqBuf);

    size_t global = N;

    /* ===== kernel futtatás ===== */
    clEnqueueNDRangeKernel(
        queue,
        kernel,
        1,
        NULL,
        &global,
        NULL,
        0,
        NULL,
        NULL);

    clFinish(queue);

    /* ===== eredmény vissza ===== */
    clEnqueueReadBuffer(
        queue,
        freqBuf,
        CL_TRUE,
        0,
        sizeof(int)*(MAX_VALUE+1),
        freq,
        0,NULL,NULL);

    /* ===== kiírás ===== */
    printf("Ertek -> gyakorisag\n");
    for(int i=0;i<=MAX_VALUE;i++)
        if(freq[i] > 0)
            printf("%d -> %d\n", i, freq[i]);

    /* ===== cleanup ===== */
    free(source);
    clReleaseMemObject(dataBuf);
    clReleaseMemObject(freqBuf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}