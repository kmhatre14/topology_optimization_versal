#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>
#include "nn.hpp"



image_3D::image_3D(){}
image_3D::image_3D(int channels, int height, int width) : img(channels,tensor_bf_2D(height,tensor_bf_1D(width,0))){}
image_3D::image_3D(int channels, int height, int width, int val) : img(channels,tensor_bf_2D(height,tensor_bf_1D(width,val))){}
image_3D::~image_3D(){}

inline int image_3D::get_conv_C()
{
    return this->img.size();
}
inline int image_3D::get_conv_H()
{
    return this->img[0].size();
}
inline int image_3D::get_conv_W()
{
    return this->img[0][0].size();
}

/* Convolution Layer*/
conv::conv()
{
    this->stride = 1;
    this->padding = 0;
}
conv::conv(int N, int channels, int height, int width) : layer(N,image_3D(channels,height,width))
{
    this->stride = 1;
    this->padding = 0;
}
conv::conv(int N, int channels, int height, int width, int val) : layer(N,image_3D(channels,height,width, val))
{
    this->stride = 1;
    this->padding = 0;
}
conv::conv(int N, int channels, int height, int width, int stride, int padding) : layer(N,image_3D(channels,height,width))
{
    this->stride = stride;
    this->padding = padding;
}
conv::~conv(){}

inline int conv::get_conv_N()
{
    return this->layer.size();
}
inline int conv::get_conv_C()
{
    return this->layer[0].get_conv_C();
}
inline int conv::get_conv_H()
{
    return this->layer[0].get_conv_H();
}
inline int conv::get_conv_W()
{
    return this->layer[0].get_conv_W();
}

/* Neural network */
nn::nn()
{

}
nn::nn(conv Layer) : Layer(Layer.get_conv_N(), Layer.get_conv_C(), Layer.get_conv_H(), Layer.get_conv_W())
{

}
nn::~nn()
{

}

// flattens in row major order
void nn::conv2mat(const conv vec4d, mat*const out) {
    if (vec4d.layer.empty() || vec4d.layer[0].img.empty() || vec4d.layer[0].img[0].empty() || vec4d.layer[0].img[0][0].empty())
    {
        return;
    }

    int K = vec4d.layer.size();
    int C = vec4d.layer[0].img.size();
    int R = vec4d.layer[0].img[0].size();
    int S = vec4d.layer[0].img[0][0].size();

    out->mat_alloc(K, C*R*S);

    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < C; ++c) {
            for (int r = 0; r < R; ++r) {
                for (int s = 0; s < S; ++s) {
                    int col = r*S + c * S* R + s;
                    out->mat_at(k, col) = vec4d.layer[k].img[c][r][s];
                }
            }
        }
    }
}

void nn::mat_mul_elementwise(mat input1, mat input2, mat*const output)
{
    input1.mat_dot(input2,output);
}

// A(16 x 9) B(9 x 36000)
// M(96 x 64) N(64 x 96)
void nn::gemm(mat A, mat B, mat* const C)
{
    tensor_bf_4D N_tiles;
    tensor_bf_4D M_tiles;

    mat M(M_tile_size[0], M_tile_size[1]);
    mat N(N_tile_size[0], N_tile_size[1]);
    mat out(M_tile_size[0], N_tile_size[1]);

    N_tiles = nn::mattiled(B, N_tile_size[0], N_tile_size[1]);
    M_tiles = nn::mattiled(A, M_tile_size[0], M_tile_size[1]);

    A.print_2d_dim("A: ");
    B.print_2d_dim("B: ");

    // Dimension check
    if (B.mat_row() != A.mat_col())
    {
        std::cout << "Dimension miss match" << std::endl;
        std::cout << "B_K:" << B.mat_row() << std::endl;
        std::cout << "A_K:" << A.mat_col() << std::endl;
        return;
    }
    else if (M_tiles[0].size() != N_tiles.size())
    {
        std::cout << "Tile Dimension miss match" << std::endl;
        std::cout << "M_K:" << M_tiles.size() << std::endl;
        std::cout << "N_K:" << N_tiles[0].size() << std::endl;
        return;
    }

    C->mat_alloc(A.mat_row(), B.mat_col());

    for(int i = 0; i<M_tiles.size(); i++)
    {
        for(int j = 0; j<N_tiles[0].size(); j++)
        {
            for(int k = 0; k<M_tiles[0].size(); k++)
            {
                M.tensor2D2mat(M_tiles[i][k]);
                N.tensor2D2mat(N_tiles[i][k]);
                M.mat_mul(N,&out);
                
                // Mering into one output matrix
                for (int pos_i = 0; pos_i < M_tile_size[0] ; pos_i++)
                {
                    for(int pos_j = 0; pos_j < N_tile_size[1] ; pos_j++)
                    {
                        int row = i*M_tiles.size()+ pos_i;
                        int col = j*N_tiles[0].size() + pos_j;
                        //std::cout << "No issue here" << std::endl;
                        C->mat_at(row,col) = out.mat_at(pos_i,pos_j);
                    }
                }
            }
        }
    }
}

void nn::gemm2conv(mat input, conv*const output, int N, int F, int OH, int OW)
{
    // Validate total elements
    if (input.mat_row() != N * F * OH * OW)
    {
        std::cout<< "Dimession mismatch:" << std::endl;
        std::cout<< "input rows: " << input.mat_row() << std::endl;
        std::cout<< "F x N x OH x OW: " << std::endl;
        return;
    }

    // Calculate stride values for index computation
    const int stride_F = N * OH * OW;
    const int stride_N = OH * OW;
    const int stride_OH = OW;

    // Single-pass construction with merged reshape+transpose
    for (int n = 0; n < N; ++n) {
        for (int f = 0; f < F; ++f) {
            for (int h = 0; h < OH; ++h) {
                for (int w = 0; w < OW; ++w) {
                    // Calculate original index in flat_data
                    const int src_idx = f * stride_F    // Filter dimension
                                      + n * stride_N    // Batch dimension
                                      + h * stride_OH;   // Height dimension

                    // Direct assignment to final transposed position
                    output->layer[n].img[f][h][w] = input.mat_at(src_idx,w);
                }
            }
        }
    }
}

void nn::convolution(conv input, conv kernel, conv bias, int stride, int padding, conv* output)
{
    mat mat_im2col;
    mat mmul;
    mat kernel_flat;
    mat bias_flat;
    mat out;
    int OH = (input.get_conv_H() + 2*padding - kernel.get_conv_H())/stride + 1;
    int OW = (input.get_conv_W() + 2*padding - kernel.get_conv_W())/stride + 1;
    conv conv_out(input.get_conv_N(),kernel.get_conv_C(),OH,OW);

    nn::conv2mat(kernel, &kernel_flat);
    nn::im2col(input, kernel, stride, padding, &mat_im2col);
    nn::gemm(kernel_flat, mat_im2col, &mmul);
    mmul.mat_add(bias_flat, &out);
    gemm2conv(out, &conv_out, input.get_conv_N(), kernel.get_conv_C(), OH,OW);

    output = &conv_out;
}

void nn::convolution(conv input, conv kernel, conv bias, int stride, conv* const output)
{
    this->convolution(input,kernel,bias,stride,0,output);
}

// Applies the sigmoid function elementwise to an input array
inline bfloat16 nn::sigmoid_func(bfloat16 x) {
    return 1.0f / (1.0f + exp(-x));
}

void nn::sigmoid(conv conv_output, conv output)
{
    int out_channels = conv_output.get_conv_C();
    int out_h = conv_output.get_conv_H();
    int out_w = conv_output.get_conv_W();
    int out_n = conv_output.get_conv_N();

    for(int n=0; n < out_n; n++)
    {
        for (int c = 0; c < out_channels; ++c) {
            for (int i = 0; i < out_h; ++i) {
                for (int j = 0; j < out_w; ++j)
                {
                    output.layer[n].img[c][i][j] = sigmoid_func(conv_output.layer[n].img[c][i][j]);
                }
                
            }
        }
    }     
}
/*
// Adaptive max pooling for 3D input: [channels, height, width]
// For NCHW format data:
// input_dims = [channels, height, width]
// output_dims = [channels, new_height, new_width]
void adaptive_max_pool(const vector<float>& input, vector<float>& output,
    int in_channels, int in_h, int in_w,
    int out_h, int out_w) 
    {
    output.resize(in_channels * out_h * out_w);

    #pragma omp parallel for
    for(int c = 0; c < in_channels; ++c) {
        for(int oh = 0; oh < out_h; ++oh) {
            int h_start = (oh * in_h) / out_h;
            int h_end = ((oh + 1) * in_h) / out_h;

            for(int ow = 0; ow < out_w; ++ow) {
                int w_start = (ow * in_w) / out_w;
                int w_end = ((ow + 1) * in_w) / out_w;

                float max_val = numeric_limits<float>::lowest();
                for(int h = h_start; h < h_end; ++h) {
                    for(int w = w_start; w < w_end; ++w) {
                        int idx = c*in_h*in_w + h*in_w + w;
                        max_val = max(max_val, input[idx]);
                    }
                }
                output[c*out_h*out_w + oh*out_w + ow] = max_val;
            }
        }
    }
}
*/
tensor_bf_4D nn::mattiled(mat A, int tile_r, int tile_c)
{
    int num_tile_r;
    int num_tile_c;
    int r;
    int c;
    int H = A.mat_row();
    int W = A.mat_col();

    num_tile_r = (H % tile_r == 0)? (H / tile_r): (H / tile_r + 1);
    num_tile_c = (W % tile_c == 0)? (W / tile_c):  (W / tile_c + 1);
    
    std::cout << "no. of tile_r: " << num_tile_r << std::endl;
    std::cout << "no of tile_c: " << num_tile_c << std::endl;

    tensor_bf_4D final_tiles(num_tile_r, tensor_bf_3D(num_tile_c, tensor_bf_2D(tile_r, tensor_bf_1D(tile_c))));

    for (int h = 0 ; h < num_tile_r; h++)
    {
        for (int w = 0; w < num_tile_c; w++)
        {
            for (int i=0; i<tile_r ; i++)
            {
                for (int j=0; j<tile_c; j++)
                {
                    r = h*tile_r + i;
                    c = w*tile_c + j;
                    if ((r >= H) || (c >= W))
                    {
                        final_tiles[h][w][i][j] = 0;
                    }
                    else
                    {
                        final_tiles[h][w][i][j] = A.mat_at(r,c);
                    }
                }
            }
        }
    }

    //print_4d_matrix(final_tiles, "Final tiles");
    return final_tiles;
}

void nn::im2col(conv input, conv kernel, int stride, int padding, mat* const output) {
    int in_channels = input.get_conv_C();
    int out_channels = kernel.get_conv_C();
    if (in_channels != out_channels)
    {
        std::cout<< " Input kernel channel missmatch "<< std::endl;
        std::cout<< " Input channel "<< in_channels << std::endl;
        std::cout<< " output channel "<< out_channels << std::endl;
        return;
    }
    int pad = padding;
    int N = input.get_conv_N();
    int C = input.get_conv_C();
    int H = input.get_conv_H();
    int W = input.get_conv_W();

    int KH = kernel.get_conv_H();
    int KW = kernel.get_conv_W();

    int OH = (H - KH + 2 * pad) / stride + 1;
    int OW = (W - KW + 2 * pad) / stride + 1;

    // Pad input if needed
    auto input_padded = pad_input(input, pad);
    output->mat_alloc(KH * KW * C, N * OH * OW);
    
    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                int col_index = n * OH * OW + oh * OW + ow;
                for (int c = 0; c < C; ++c) {
                    for (int kh = 0; kh < KH; ++kh) {
                        for (int kw = 0; kw < KW; ++kw) {
                            int h_index = oh * stride + kh;
                            int w_index = ow * stride + kw;
                            int row_index = c * KH * KW + kh * KW + kw;
                            output->mat_at(row_index, col_index) = input_padded.layer[n].img[c][h_index][w_index];
                        }
                    }
                }
            }
        }
    }
}

conv nn::pad_input(conv X, int pad) 
{
    if (pad == 0) return X;

    int N = X.get_conv_N();
    int C = X.get_conv_C();
    int H = X.get_conv_H();
    int W = X.get_conv_W();

    int H_padded = H + 2 * pad;
    int W_padded = W + 2 * pad;

    conv X_padded(N,C, H_padded, W_padded);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H_padded; ++h) {
                for (int w = 0; w < W_padded; ++w) {
                    if((h > H) || (h < pad) || (w > W) || (w < pad))
                    {
                        X_padded.layer[n].img[c][h][w] = 0;
                    }
                    else
                    {
                        X_padded.layer[n].img[c][h][w] = X.layer[n].img[c][h-pad][w-pad];
                    }
                }
            }
        }
    }
    return X_padded;
}

void nn::print_conv_dim(std::string st, conv layer)
{
    std::cout << st << std::endl;
    std::cout<< " N : " << layer.get_conv_N() << std::endl;
    std::cout<< " C : " << layer.get_conv_C() << std::endl;
    std::cout<< " H : " << layer.get_conv_H() << std::endl;
    std::cout<< " W : " << layer.get_conv_W() << std::endl;
}