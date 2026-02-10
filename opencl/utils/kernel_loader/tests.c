#include "kernel_loader.h"

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <setjmp.h>
#include <cmocka.h>

#include <stdlib.h>

static void test_empty_file(void **state)
{
    char* source_code;
    int error_code;

    source_code = load_kernel_source("kernels/empty.cl", &error_code);

    assert_non_null(source_code);
    assert_string_equal(source_code, "");
    assert_int_equal(error_code, 0);

    if (source_code != NULL) {
        free(source_code);
    }
}

static void test_single_line(void **state)
{
    char* source_code;
    int error_code;

    source_code = load_kernel_source("kernels/single_line.cl", &error_code);

    assert_non_null(source_code);
    assert_string_equal(source_code, "Single line here!");
    assert_int_equal(error_code, 0);

    if (source_code != NULL) {
        free(source_code);
    }
}

static void test_multiple_line(void **state)
{
    char* source_code;
    int error_code;

    source_code = load_kernel_source("kernels/multiple_line.cl", &error_code);

    assert_non_null(source_code);
    assert_string_equal(source_code, "Multiple\nline\nhere!\n");
    assert_int_equal(error_code, 0);

    if (source_code != NULL) {
        free(source_code);
    }
}

static void test_invalid_path(void **state)
{
    char* source_code;
    int error_code;

    source_code = load_kernel_source("kernels/missing.cl", &error_code);

    assert_null(source_code);
    assert_int_equal(error_code, -1);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_empty_file),
        cmocka_unit_test(test_single_line),
        cmocka_unit_test(test_multiple_line),
        cmocka_unit_test(test_invalid_path)
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
