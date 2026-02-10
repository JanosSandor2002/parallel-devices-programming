#include <stdio.h>
#include <CL/cl.h>

int main() {
    cl_uint num_platforms;
    cl_int err;

    // Get number of OpenCL platforms
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        printf("No OpenCL platforms found.\n");
        return 1;
    }

    cl_platform_id platforms[num_platforms];
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        printf("Failed to get OpenCL platforms.\n");
        return 1;
    }

    printf("Found %u OpenCL platform(s):\n", num_platforms);
    for (cl_uint i = 0; i < num_platforms; i++) {
        char name[128];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(name), name, NULL);
        printf("Platform %u: %s\n", i, name);

        // List devices for this platform
        cl_uint num_devices;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        cl_device_id devices[num_devices];
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

        for (cl_uint j = 0; j < num_devices; j++) {
            char device_name[128];
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
            printf("  Device %u: %s\n", j, device_name);
        }
    }

    return 0;
}
