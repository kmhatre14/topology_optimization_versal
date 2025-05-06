/*
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
*/
#ifndef __KERNEL_H__
#define __KERNEL_H__
#include <adf.h>
using namespace adf;
void weighted_sum_with_margin(input_buffer<int32, extents<256>, margin<8>> &in,
                              output_buffer<int32, extents<256>> &out);
#endif
