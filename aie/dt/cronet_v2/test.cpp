#include <iostream>
#include <fstream>
#include "data_types.hpp"

//#include "mat.hpp"
#include "nn.hpp"

//#define MAT_TEST
#define NN_TEST

int main ()
{
    #ifdef MAT_TEST
        int M = 10;
        int K = 5;
        int N = 10;

        mat A(M,K), B(K,N), C(M,N);
        A.set_matrix_val(1);
        B.set_matrix_val(2);

        // matrix multiplication
        A.print_2d_mem("A:");
        B.print_2d_mem("B:");
        A.mat_mul(B, &C);
        C.print_2d_mem("C:");

        // matrix copy
        mat D(M,N);
        D.print_2d_mem("before cpy:");
        if (D.matcpy(C) == FUNC_NOK)
        {
            std::cout << "dimenssion mismatch" << std::endl;
        }
        D.print_2d_mem("after cpy:");

        // matrix transpose
        A.print_2d_mem("before cpy:");
        A.transpose();
        A.print_2d_mem("after cpy:");
    #endif

    #ifdef NN_TEST
        int N = 3;
        int C = 3;
        int H = 10;
        int W = 20;
        conv input(N,C,H,W,1);
        conv bias(N,C,H,W);

        int kR = 2;
        int kS = 2;
        int kC = 3;
        int K = 10; 
        conv kernel(K,kC,kR,kS);
        
        conv* output = nullptr;
        nn trunknet;
        //trunknet.print_conv_dim("Input:",input);
        //trunknet.print_conv_dim("Kernel:", kernel);
        trunknet.convolution(input,kernel,bias,1,0,output);

    #endif

    return 0;
}