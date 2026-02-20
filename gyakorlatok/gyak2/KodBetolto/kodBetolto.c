#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

/* Kernel fajl teljes beolvasasa */
char* readKernelSource(const char* filename, size_t* size_out) {

    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Nem sikerult megnyitni: %s\n", filename);
        return NULL;
    }

    /* fajl vege -> meret */
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    rewind(file);

    /* +1 a '\0' miatt */
    char* source = (char*)malloc(size + 1);
    if (!source) {
        printf("Memoriafoglalasi hiba\n");
        fclose(file);
        return NULL;
    }

    fread(source, 1, size, file);
    source[size] = '\0';   // string lezárás

    fclose(file);

    if (size_out)
        *size_out = size;

    return source;
}

int main() {

    size_t kernel_size;
    char* kernel_source =
        readKernelSource("kernel.cl", &kernel_size);

    if (!kernel_source)
        return 1;

    printf("Kernel sikeresen beolvasva!\n");
    printf("Meret: %zu byte\n\n", kernel_size);

    printf("=== Kernel tartalma ===\n");
    printf("%s\n", kernel_source);

    free(kernel_source);
    return 0;
}