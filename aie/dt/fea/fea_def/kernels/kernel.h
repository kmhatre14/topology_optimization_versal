#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H
#include <adf.h>

  void FE_kernel(
    adf::input_buffer<float> & x_in,
    adf::output_buffer<float> & U_out,
    adf::output_buffer<float> & K_out,
    adf::output_buffer<float> & F_out,
    adf::output_buffer<float> & KE_out
  );
#endif
