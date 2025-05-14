#include <iostream>
#include <fstream>

//#include "mat.hpp"
#include "nn.hpp"

//#define MAT_TEST
#define NN_TEST

mm_graph mm_graph_0;

mat::mat()
{
    this->data.rows = 0;
    this->data.cols = 0;
    this->data.size = 0;
    this->data.arr_2d = nullptr;
}

mat::mat(int row, int col)
{
    this->data.rows = row;
    this->data.cols = col;
    this->data.size = row*col;
    #ifdef AMD_VIRSAL
    this->data.arr_2d = (bfloat16 *)GMIO::malloc(row * col * sizeof(bfloat16));
    #else
    this->data.arr_2d = (bfloat16 *)malloc(row * col * sizeof(bfloat16));
    #endif
}

mat::~mat()
{
    #ifdef AMD_VIRSAL
    //GMIO::free(this->data.arr_2d);
    //std::cout << "GMIO transactions finished" << std::endl;
    #endif
    this->data.arr_2d = nullptr;
    this->data.rows = 0;
    this->data.cols = 0;
    this->data.size = 0;
}

FUNC_RET mat::mat_alloc(int rows, int cols)
{
    if (this->data.arr_2d == nullptr)
    {
        #ifdef AMD_VIRSAL
        this->data.arr_2d = (bfloat16 *)GMIO::malloc(rows * cols * sizeof(bfloat16));
        #else
        this->data.arr_2d = (bfloat16 *)malloc(rows * cols * sizeof(bfloat16));
        #endif
        this->data.rows = rows;
        this->data.cols = cols;
        this->data.size = rows*cols;
        return FUNC_OK;
    }
    else
    {
        return FUNC_NOK;
    }
}

FUNC_RET mat::matcpy( mat M)
{
    if ((this->data.rows != M.data.rows) || (this->data.cols != M.data.cols) \
    || (this->data.arr_2d == nullptr) || (M.data.arr_2d == nullptr))
    {
        return FUNC_NOK;
    }

    int rows = this->data.rows;
    int cols = this->data.cols;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            this->mat_at(i,j) = M.mat_at(i,j);
        }
    }

    return FUNC_OK;
}
bool mat::mat_is_alloc()
{
    if ((this->data.arr_2d == nullptr))
    {
        return false;
    }
    else
    {
        return true;
    }
}
int mat::mat_row()
{
    int ret = this->data.rows;
    return ret;
}

int mat::mat_col()
{
    return this->data.cols;
}

bfloat16& mat::mat_at(int x, int y)
{
    return this->data.arr_2d[x*this->data.cols + y];
}

bfloat16 mat::mat_at(int x, int y) const
{
    return this->data.arr_2d[x*this->data.cols + y];
}

void mat::set_matrix_val(int value) 
{
    if (this->data.arr_2d == nullptr)
    {
        #ifdef AMD_VIRSAL
        this->data.arr_2d = (bfloat16 *)GMIO::malloc(this->data.rows * this->data.cols * sizeof(bfloat16));
        #else
        this->data.arr_2d = (bfloat16 *)malloc(this->data.rows * this->data.cols * sizeof(bfloat16));
        #endif
    }
    // fill up with values
    for (int i = 0; i < this->data.rows; i++) {
        for (int j = 0; j < this->data.cols; j++) {
            this->mat_at(i,j) =value;
        }
    }
}

void mat::print_2d_mem(std::string st) 
{
    std::cout << st << ":" << std::endl;
    for (int i = 0; i < this->data.rows; i++) {
        for (int j = 0; j < this->data.cols; j++) {
            std::cout << this->mat_at(i,j) << " ";
        }
        std::cout << std::endl;
    }
}

void mat::print_2d_dim(std::string st)
{
    std::cout << st << std::endl;
    std::cout << " Rows: " <<this->data.rows << std::endl;
    std::cout << " Cols: " <<this->data.cols << std::endl;
}

FUNC_RET mat::mat_add(mat M, mat*const out)
{
    if ((this->data.rows != M.data.rows) || (this->data.cols != M.data.cols) || (out == nullptr))
    {
        return FUNC_NOK;
    }

    int rows = this->data.rows;
    int cols = this->data.cols;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            out->mat_at(i,j) = this->mat_at(i,j) + M.mat_at(i,j);
        }
    }

    return FUNC_OK; 
}

FUNC_RET mat::mat_mul(mat N, mat*const output)
{
    if (this->data.cols != N.data.rows)
    {
        std::cout << "Error: Dimension miss match" << std::endl;
        std::cout << "M_K:" << this->data.cols << std::endl;
        std::cout << "N_K:" << N.data.rows << std::endl;
        return FUNC_NOK;
    }
    else if((output->data.rows != this->data.rows) || (output->data.cols != N.data.cols))
    {
        std::cout << "Error: Dimension miss match" << std::endl;
        std::cout << "output_rows:" << output->data.rows << std::endl;
        std::cout << "output_cols:" << output->data.cols << std::endl;
        std::cout << "M_rows:" << this->data.rows << std::endl;
        std::cout << "N_cols:" << N.data.cols << std::endl;
        return FUNC_NOK;
    }

    if ((output == nullptr) || (output->data.arr_2d == nullptr) || (output->data.size == 0))
    {
        std::cout<<"Error: matrix not allocated" << std::endl;
        return FUNC_NOK;
    }

    this->print_2d_dim("M dim:");
    N.print_2d_dim("N dim:");
    output->print_2d_dim("Output dim:");

    #ifdef AMD_VIRSAL
    std::cout << "MMUL started!!" << std::endl;
    mm_graph_0.in_lhs[0].gm2aie(this->data.arr_2d, sizeof(this->data.arr_2d[0]) * this->data.rows * this->data.cols);
    mm_graph_0.in_rhs[0].gm2aie(N.data.arr_2d, sizeof(N.data.arr_2d[0]) * N.mat_row() * N.mat_col());
    mm_graph_0.run(1);
    mm_graph_0.out[0].aie2gm(output->data.arr_2d,sizeof(output->data.arr_2d[0]) * output->data.rows * output->data.cols);
    std::cout << "MMUL ended!!" << std::endl;
    #else
    bfloat16 sum = 0;
    for(int i = 0; i<this->data.rows; i++)
    {
        for(int j = 0; j< N.data.cols; j++)
        {
            sum = 0;
            for(int k = 0; k< this->data.cols; k++)
            {
                sum += this->mat_at(i,k) * N.mat_at(k,j);
            }
            out->mat_at(i,j) = sum;
        }
    }
    #endif
    return FUNC_OK;
}

FUNC_RET mat::mat_dot(mat N, mat*const out)
{
    if (this->data.cols != N.data.rows)
    {
        std::cout << "M_K:" << this->data.rows << std::endl;
        std::cout << "N_K:" << N.data.cols << std::endl;
        std::cout << "Error: Dimension miss match" << std::endl;
        return FUNC_NOK;
    }

    if ((out == nullptr) || (out->data.arr_2d == nullptr) || (out->data.size == 0))
    {
        std::cout<<"Error: matrix not allocated" << std::endl;
        return FUNC_NOK;
    }
    
    bfloat16 sum = 0;
    for(int i = 0; i<this->data.rows; i++)
    {
        for(int j = 0; j< N.data.cols; j++)
        {
           out->mat_at(i,j) = this->mat_at(i,j) * N.mat_at(i,j);
        }
    }
    return FUNC_OK;
}

void mat::transpose() 
{
    mat result(this->data.cols, this->data.rows);

    for (int i = 0; i < result.data.rows; ++i) {
        for (int j = 0; j < result.data.cols; ++j) {
            result.mat_at(i,j) = this->mat_at(j,i);
        }
    }
    this->data.arr_2d = result.data.arr_2d;
    this->data.rows = result.data.rows;
    this->data.cols = result.data.cols;
}

FUNC_RET mat::tensor2D2mat(tensor_bf_2D tensor)
{
    if ((this->data.rows != tensor.size()) || (this->data.cols != tensor[0].size()))
    {
        std::cout << "Dimension miss match" << std::endl;
        std::cout << "mat row: " << this->data.rows << std::endl;
        std::cout << "tensor row: " << tensor.size() << std::endl;
        std::cout << "mat cols: " << this->data.cols << std::endl;
        std::cout << "tensor cols: " << tensor[0].size() << std::endl;
        return FUNC_NOK;
    }

    for (int i = 0; i < this->data.rows; ++i) {
        for (int j = 0; j < this->data.cols; ++j) {
            this->mat_at(i,j) = tensor[i][j];
        }
    }
    return FUNC_OK;
}

/* Neural Networks */
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

void print_2d_matrix(tensor_bf_2D A, std::string name) {
    std::cout << name << std::endl;
    for (int i = 0; i < A.size(); i++) {
        std::cout << "[";
        for (int j = 0; j < A[0].size(); j++) {
            std::cout << A[i][j] << "   ";
        }
        std::cout << "]" << std::endl;
    }
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
    out.set_matrix_val(0);

    M_tiles = nn::mattiled(A, M_tile_size[0], M_tile_size[1]);
    N_tiles = nn::mattiled(B, N_tile_size[0], N_tile_size[1]);

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
        std::cout << "M_K:" << M_tiles[0].size() << std::endl;
        std::cout << "N_K:" << N_tiles.size() << std::endl;
        return;
    }

    C->mat_alloc(A.mat_row(), B.mat_col());
    mat sum(M_tile_size[0],N_tile_size[1]);
    sum.set_matrix_val(0);
    
    std::vector<std::vector<mat>> temp(N_tiles.size(), std::vector<mat>(M_tiles[0].size(), mat(M_tile_size[0],N_tile_size[1])));
    
    for(int i = 0; i<M_tiles.size(); i++)
    {
        for(int j = 0; j<N_tiles[0].size(); j++)
        {
            for(int k = 0; k<M_tiles[0].size(); k++)
            {
                M.tensor2D2mat(M_tiles[i][k]);
                N.tensor2D2mat(N_tiles[k][j]);
                //M.print_2d_dim("M tile size:");
                //N.print_2d_dim("N tile size:");
                //M.print_2d_mem("M data:");
                //N.print_2d_mem("N data:");
                out.print_2d_dim("Out:");
                //out.print_2d_mem("Out data:");
                M.mat_mul(N,&out);
                std::cout << "multiplication done" << std::endl;
                sum.mat_add(out,&sum);
                std::cout << "accumulated done" << std::endl;
            }
            
            int ret = temp[i][j].matcpy(sum);
            sum.set_matrix_val(0);
        }
    }
    
    // Mering into one output matrix
    int row =0;
    int col =0;
    for (int i = 0; i < temp.size() ; i++)
    {
        for(int j = 0; j < temp[0].size() ; j++)
        {
            for(int pos_i = 0; pos_i<temp[0][0].mat_row(); pos_i++)
            {
                for (int pos_j = 0; pos_j < temp[0][0].mat_col(); pos_j++)
                {

                    if (row >= A.mat_row())
                    {
                        row = A.mat_row();
                    }
                    else if (col >= B.mat_col())
                    {
                        col = B.mat_col();
                    }
                    else
                    {
                        C->mat_at(row,col) = temp[i][j].mat_at(pos_i, pos_j);
                        row = i*B.mat_col();
                        col = j*N_tile_size[1] + pos_j;
                    }
                }
            }
        }
    }
    C->print_2d_dim("C:");
}

void nn::gemm2conv(mat input, conv*const output, int N, int K, int OH, int OW)
{
    // Validate total elements
    if ((input.mat_col() != N * OH * OW) && (input.mat_row() != K)) 
    {
        std::cout<< "Dimession mismatch:" << std::endl;
        std::cout<< "input rows: " << input.mat_row() << std::endl;
        std::cout<< "input cols: " << input.mat_col() << std::endl;
        std::cout<< "K: " << K << std::endl;
        std::cout<< "N x OH x OW: " << (N * OH * OW) << std::endl;
        return;
    }

    // Calculate stride values for index computation
    const int stride_K = N * OH * OW;
    const int stride_N = OH * OW;
    const int stride_OH = OW;

    // Single-pass construction with merged reshape+transpose
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int h = 0; h < OH; ++h) {
                for (int w = 0; w < OW; ++w) {
                    // Calculate original index in flat_data
                    const int src_idx = n * stride_N    // Batch dimension
                                      + h * stride_OH + w;   // Height dimension
                    // Direct assignment to final transposed position
                    output->layer[n].img[k][h][w] = input.mat_at(k,src_idx);
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
    int OH = (input.get_conv_H() - kernel.get_conv_H()) + 1;
    int OW = (input.get_conv_W() - kernel.get_conv_W()) + 1;

    nn::conv2mat(kernel, &kernel_flat);
    nn::conv2mat(bias, &bias_flat);
    nn::im2col(input, kernel, stride, padding, &mat_im2col);
    nn::gemm(kernel_flat, mat_im2col, &mmul);
    gemm2conv(mmul, output, input.get_conv_N(), kernel.get_conv_N(), OH,OW);
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
// Kor NCHW format data:
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

    //print_4d_matrix(final_tiles, "Kinal tiles");
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
    //print_conv_dim("input:", input);
    //print_conv_dim("kernel", kernel);
    //std::cout<< " OH: " << OH << std::endl;
    //std::cout << " OW: " << OW << std::endl;
    // Pad input if needed
    conv input_padded = pad_input(input, pad);
    output->mat_alloc(KH * KW * C, N * OH * OW);
    //std::cout<< " O row: " << KH * KW * C << std::endl;
    //std::cout << " O col: " << N * OH * OW << std::endl;
    
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

#if defined(__AIESIM__) || defined(__X86SIM__)
int main(int argc, char **argv)
{

    #ifdef MAT_TEST
        int M = 64;
        int K = 96;
        int N = 64;

        mm_graph_0.init();
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
        int N = 30;
        int C = 1;
        int H = 20;
        int W = 60;
        conv input(N,C,H,W,1);
        conv bias(N,C,H,W);

        int kR = 3;
        int kS = 3;
        int kC = 1;
        int K = 16; 
        conv kernel(K,kC,kR,kS,1);
        
        int OH = (input.get_conv_H() - kernel.get_conv_H()) + 1;
        int OW = (input.get_conv_W() - kernel.get_conv_W()) + 1;
        conv output(input.get_conv_N(),kernel.get_conv_N(),OH,OW);
        nn trunknet;
        //trunknet.print_conv_dim("Input:",input);
        //trunknet.print_conv_dim("Kernel:", kernel);
        trunknet.convolution(input,kernel,bias,1,1,&output);
        trunknet.print_conv_dim("Conv Output:", output);

    #endif

    return 0;
}

#endif