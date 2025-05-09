#include <vector>
#include <fstream>
#include "kernels/aie_top.h"
#include "nn.hpp"
using namespace adf;

typedef std::vector<std::vector<std::vector<std::vector<bfloat16>>>>tensor_bf_4D; // 4D tensor type definition
typedef std::vector<std::vector<std::vector<bfloat16>>>tensor_bf_3D;                                     // 3D tensor type definition
typedef std::vector<std::vector<bfloat16>> tensor_bf_2D; // 2D tensor type definition
typedef std::vector<bfloat16> tensor_bf_1D;              // 1D tensor type definition

/* Function declearation */
tensor_bf_4D mattiled(tensor_bf_2D A, int tile_r, int tile_c);
bfloat16 *generate_2D_matrix(int row, int col, int value);
int get_matrix_file(std::string file, int row, int col, tensor_bf_2D* const A);
void get_matrix_val(tensor_bf_2D * const A, bfloat16 val);
void print_2d_matrix(tensor_bf_2D A, std::string name);
void print_2d_mem(std::string st, bfloat16 *A, int size[2]);

/* Global variables */
mm_graph mm_graph0;

tensor_bf_4D mattiled(tensor_bf_2D A, int tile_r, int tile_c) {
    int num_tile_r;
    int num_tile_c;
    int r;
    int c;
    int H = A.size();
    int W = A[0].size();

    num_tile_r = (H % tile_r == 0) ? (H / tile_r) : (H / tile_r + 1);
    num_tile_c = (W % tile_c == 0) ? (W / tile_c) : (W / tile_c + 1);

    std::cout << num_tile_r << std::endl;
    std::cout << num_tile_c << std::endl;

    tensor_bf_4D final_tiles(num_tile_r, tensor_bf_3D(num_tile_c, tensor_bf_2D(tile_r, tensor_bf_1D(tile_c))));

    for (int h = 0; h < num_tile_r; h++) {
        for (int w = 0; w < num_tile_c; w++) {
            for (int i = 0; i < tile_r; i++) {
                for (int j = 0; j < tile_c; j++) {
                    r = h * tile_r + i;
                    c = w * tile_c + j;
                    if ((r >= H) || (c >= W)) {
                        final_tiles[h][w][i][j] = 0;
                    } else {
                        final_tiles[h][w][i][j] = A[r][c];
                    }
                }
            }
        }
    }

    // print_4d_matrix(final_tiles, "Final tiles");
    return final_tiles;
}

int get_matrix_file(std::string file, int row, int col, tensor_bf_2D* const A)
{
    int number;

    // input0 file
    std::ifstream inputFile("data/" + file);
    if (!inputFile.is_open()) {
        std::cerr << "Error opening the file!" << std::endl;
        return 1;
    }

    while (inputFile >> number) {
        for(int i = 0; i<row; i++)
        {
            for(int j = 0; j<col; j++)
            {
                (*A)[i][j] = number;
            }
        }
    }
    return 0;
}

void get_matrix_val(tensor_bf_2D * const A, bfloat16 val) {
    for(int i = 0; i< A->size(); i++)
    {
        for(int j = 0; j< (*A)[0].size(); j++)
        {
            (*A)[i][j] = val;
        }
    }
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

void print_4d_matrix (tensor_bf_4D A, std::string name)
{
    std::cout << name << std::endl;
    for (int i = 0; i<A.size(); i++)
    {
        std:: cout << "[";
        for(int j = 0; j<A[0].size(); j++)
        {
            print_2d_matrix(A[i][j],"");
        }
        std:: cout << "]" << std::endl;
    }
}

bfloat16 *generate_2D_matrix(int row, int col, int value) {
    bfloat16 *arr = (bfloat16 *)GMIO::malloc(row * col * sizeof(bfloat16));

    // fill up with values
    for (int i = 0; i < row * col; i++) {
        arr[i] = value;
    }

    return arr;
}

void print_2d_mem(std::string st, bfloat16 *A, int size[2]) {
    std::cout << st << ":" << std::endl;
    int pos = 0;
    for (int i = 0; i < size[0]; i++) {
        for (int j = 0; j < size[1]; j++) {
            std::cout << A[pos++] << " ";
        }
        std::cout << std::endl;
    }
}

#if defined(__AIESIM__) || defined(__X86SIM__)
int main(int argc, char **argv) 
{
    // input matrix 
    int H = 9;
    int W = 36000;
    //int C = 1;
    //int N = 30;

    // weight matrix
    int kR = 16; 
    int kS = 9;
    //int kC = 1;
    //int kN = 16;

    int ret_M = 0;
    int ret_N = 0;
    int N_tile_size[2] = {L0_h1, L0_w1};
    int M_tile_size[2] = {L0_w1, L0_w2};
    int Out_tile_size[2] = {L0_h1, L0_w2};
    tensor_bf_2D M(kR, tensor_bf_1D(kS));
    tensor_bf_2D N(H, tensor_bf_1D(W));

    get_matrix_val(&M, 1);
    get_matrix_val(&N, 1);
    bfloat16 *out = generate_2D_matrix(L0_h1, L0_w2, 1);

    /*
    ret_M = get_matrix_file("input0.txt", kR, kS, &M);
    ret_N = get_matrix_file("input1.txt", H, W, &N);
    if ((ret_M == 1) || (ret_N == 1))
    {
        return 0;
    }
    */

    /*
    bfloat16 *input_M = generate_2D_matrix(L0_h1, L0_w1, 1);
    bfloat16 *input_N = generate_2D_matrix(L0_w1, L0_w2, 1);
    print_2d_mem("Matrix M", input_M, M_tile_size);
    print_2d_mem("Matrix N", input_N, N_tile_size);
    */

    std::cout << "Matrix generated!!" << std::endl;

    // Tiling input0 (L0_h1 x L0_w1) and input1 (L0_w1 x L0_w2)
    tensor_bf_4D M_tiles = mattiled(M, M_tile_size[0], M_tile_size[1]);
    tensor_bf_4D N_tiles = mattiled(N, N_tile_size[0], N_tile_size[1]);
    
    //print_4d_matrix(N_tiles, "Input Matrix");
    //print_4d_matrix(M_tiles, "Weight Matrix");

    // Dimension check
    if (M_tiles[0].size() != N_tiles.size())
    {
        std::cout << "M_K:" << M_tiles[0][0].size() << std::endl;
        std::cout << "N_K:" << N_tiles[0][0][0].size() << std::endl;
        std::cout << "Dimension miss match" << std::endl;
        return 0;
    }

    // Tiled MMUL
    std::cout << "Simulation started!!" << std::endl;
    mm_graph0.init();

    std::cout << "Kernel tile_M: " << "row: " << M_tiles.size() << " col: "<<  M_tiles[0].size() << std::endl;
    std::cout << "Kernel tile_N: " << "row: " << N_tiles.size() << " col: "<<  N_tiles[0].size() << std::endl;
    std::cout << "Kernel tile_K: " << M_tiles[0].size() << std::endl;
    for(int i = 0; i<M_tiles.size(); i++)
    {
        for(int j = 0; j<N_tiles[0].size(); j++)
        {
            for(int k = 0; k<M_tiles[0].size(); k++)
            {
                mm_graph0.in_lhs[0].gm2aie(&M_tiles[i][k][0][0], sizeof(M_tiles[i][k][0][0]) * M_tile_size[0] * M_tile_size[1]);
                mm_graph0.in_rhs[0].gm2aie(&N_tiles[k][j][0][0], sizeof(N_tiles[k][j][0][0]) * N_tile_size[0] * N_tile_size[1]);
                mm_graph0.run(1);
                mm_graph0.out[0].aie2gm(out,sizeof(*out) * Out_tile_size[0] * Out_tile_size[1]);
                //print_2d_mem("Partial Output:", out, Out_tile_size);
            }
        }
    }
    
    std::cout << "GMIO transactions finished" << std::endl;
    mm_graph0.end();

    return 0;
}
#endif