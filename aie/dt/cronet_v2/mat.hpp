
#ifndef MAT_HPP
#define MAT_HPP

#include "data_types.hpp"
#include <string>
#include <iostream>
#include <cstring>

typedef struct matrix
{
    bfloat16 *arr_2d = nullptr;
    int rows;
    int cols;
    int size;
}matrix;

class mat
{
private:
    matrix data;

public:
    // constructor
    mat();
    mat(int row, int col);
    // destructor
    ~mat();
    // get dimensions and elements
    int mat_row();
    int mat_col();
    bool mat_is_alloc();
    //writing
    bfloat16& mat_at(int x, int y);
    //reading
    bfloat16 mat_at(int x, int y) const;

    //set the matrix
    void set_matrix_val(int value);
   
    //print matrix
    void print_2d_mem(std::string st);
    void print_2d_dim(std::string st);

    // conversion
    FUNC_RET tensor2D2mat(tensor_bf_2D tensor);

    // matrix operations
    FUNC_RET mat_alloc(int row, int col);
    FUNC_RET matcpy( mat M);
    FUNC_RET mat_add(mat M, mat*const out); 
    FUNC_RET mat_mul(mat N, mat*const out);
    FUNC_RET mat_dot(mat N, mat*const out);
    void transpose();
};

#endif