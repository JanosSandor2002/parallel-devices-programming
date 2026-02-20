#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    cl_uint num_platforms;

    clGetPlatformIDs(0, NULL, &num_platforms);
    printf("Platformok szama: %u\n", num_platforms);

    cl_platform_id *platforms =
        malloc(sizeof(cl_platform_id) * num_platforms);

    clGetPlatformIDs(num_platforms, platforms, NULL);

    for (cl_uint i = 0; i < num_platforms; i++) {
        char buffer[1024];

        clGetPlatformInfo(platforms[i],
                          CL_PLATFORM_NAME,
                          sizeof(buffer),
                          buffer,
                          NULL);

        printf("Platform %u: %s\n", i, buffer);
    }

    free(platforms);
    return 0;
}