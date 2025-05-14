
#ifndef DATA_TYPES_HPP
#define DATA_TYPES_HPP

#include <vector>
#include <cstdint>
#include "kernels/aie_top.h"

#define FUNC_OK 1
#define FUNC_NOK 0

#define AMD_VIRSAL

#ifndef AMD_VIRSAL
typedef float bfloat16;
#endif

typedef std::vector<std::vector<std::vector<std::vector<bfloat16>>>> tensor_bf_4D; // 4D tensor type definition
typedef std::vector<std::vector<std::vector<bfloat16>>> tensor_bf_3D; // 3D tensor type definition
typedef std::vector<std::vector<bfloat16>> tensor_bf_2D; // 2D tensor type definition
typedef std::vector<bfloat16> tensor_bf_1D; // 1D tensor type definition

typedef uint8_t FUNC_RET;


#endif