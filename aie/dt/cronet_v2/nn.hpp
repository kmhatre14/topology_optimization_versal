
#ifndef NN_HPP
#define NN_HPP

#include "data_types.hpp"
#include "mat.hpp"

#ifndef AMD_VIRSAL
#define L0_h1 96
#define L0_w1 64
#define L0_w2 96
#endif

typedef tensor_bf_3D image;
typedef struct image_3D
{
    image img;
    // Constructor
    image_3D();
    // N C H W
    image_3D(int height, int width, int channels);
    image_3D(int height, int width, int channels, int val);
    // Destructor
    ~image_3D();
    // get 3D_image dimensions
    int get_conv_C();
    int get_conv_H();
    int get_conv_W();
}image_3D;

typedef std::vector<image_3D> convLayer;
typedef struct conv
{
    convLayer layer;
    int stride;
    int padding;
    // Constructor
    conv();
    // N C H W
    conv(int N, int channels, int height, int width);
    conv(int N, int height, int width, int channels, int val);
    conv(int N, int channels, int height, int width, int stride, int padding);
    // Destructor
    ~conv();
    // get lauyer dimensions
    int get_conv_N();
    int get_conv_C();
    int get_conv_H();
    int get_conv_W();
}conv;

class nn
{
    private:
        int M_tile_size[2] = {L0_h1, L0_w1};
        int N_tile_size[2] = {L0_w1, L0_w2};
        // convLayer to mat
        conv pad_input(conv X, int pad); 
        void im2col(conv input, conv kernel, int stride, int padding, mat* const output);
        bfloat16 sigmoid_func(bfloat16 x);
        // 3D to 2D mat
        void conv2mat(const conv vec4d, mat*const out);
        // gemm(mat) to conv
        void gemm2conv(mat input, conv*const output, int N, int channels, int height, int width);
        // Tiling for the kernel
        tensor_bf_4D mattiled(mat A, int tile_r, int tile_c);
    public:
        // Conv layers
        conv Layer;
        // Constructor 
        nn();
        nn(conv Layer);
        // Destructor
        ~nn();
        // print conv
        void print_conv_dim(std::string st, conv layer);
        // Sigmoid activation prototype
        void sigmoid(conv conv_output, conv output);
    
        // Element-wise multiplication prototype
        void mat_mul_elementwise(mat input1, mat input2, mat*const output);
    
        // Adaptive Max Pooling prototype
        void adaptive_max_pool(mat input, mat*const output);
    
        // GEMM (General Matrix Multiplication) prototype
        void gemm(mat A, mat B, mat* const C);

        // Convolution
        void convolution(conv input, conv kernel, conv bias, int stride, int padding, conv* const output);
        void convolution(conv input, conv kernel, conv bias, int stride, conv* const output);
};

#endif