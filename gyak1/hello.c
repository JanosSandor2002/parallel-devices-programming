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
        if (num_devices == 0) continue;

        cl_device_id devices[num_devices];
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

        for (cl_uint j = 0; j < num_devices; j++) {
            char device_name[128];
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
            printf("  Device %u: %s\n", j, device_name);

            // Create OpenCL context for this device
            cl_context context = clCreateContext(NULL, 1, &devices[j], NULL, NULL, &err);
            if (err != CL_SUCCESS) {
                printf("Failed to create context for device %s\n", device_name);
                continue;
            }

            // Modern command queue creation (OpenCL 3.0+)
            cl_queue_properties props[] = { 0 }; // no special properties
            cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, devices[j], props, &err);
            if (err != CL_SUCCESS) {
                printf("Failed to create command queue for device %s\n", device_name);
                clReleaseContext(context);
                continue;
            }

            printf("    Command queue created successfully.\n");

            // Cleanup
            clReleaseCommandQueue(command_queue);
            clReleaseContext(context);
        }
    }

    return 0;
}
