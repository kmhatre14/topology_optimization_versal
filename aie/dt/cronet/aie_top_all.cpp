#include <vector>
#include "kernels/aie_top.h"
using namespace adf;

mm_graph mm_graph0;

typedef std::vector<std::vector<std::vector<std::vector<float>>>>
    itensor4D; // 4D tensor type definition
typedef std::vector<std::vector<std::vector<float>>>
    itensor3D;                                     // 3D tensor type definition
typedef std::vector<std::vector<float>> itensor2D; // 2D tensor type definition
typedef std::vector<float> itensor1D;              // 1D tensor type definition

itensor4D mattiled(itensor2D A, int tile_r, int tile_c);
bfloat16 *generate_2D_matrix(int row, int col, int value);
void print_2d_matrix(itensor2D A, std::string name);
void print_2d_mem(std::string st, bfloat16 *A, int size[]);

itensor4D mattiled(itensor2D A, int tile_r, int tile_c) {
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

    itensor4D final_tiles(num_tile_r,
                          itensor3D(num_tile_c, itensor2D(tile_r, itensor1D(tile_c))));

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

bfloat16 *generate_2D_matrix(int row, int col, int value) {
    bfloat16 *arr = (bfloat16 *)GMIO::malloc(row * col * sizeof(bfloat16));

    // fill up with values
    for (int i = 0; i < row * col; i++) {
        arr[i] = value;
    }

    return arr;
}

void print_2d_matrix(itensor2D A, std::string name) {
    std::cout << name << std::endl;
    for (int i = 0; i < A.size(); i++) {
        std::cout << "[";
        for (int j = 0; j < A[0].size(); j++) {
            std::cout << A[i][j] << "   ";
        }
        std::cout << "]" << std::endl;
    }
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
int main(int argc, char **argv) {
    std::cout << "Simulation started!!" << std::endl;
    mm_graph0.init();
    int M_size[2] = {L0_h1, L0_w1};
    int N_size[2] = {L0_w1, L0_w2};

    bfloat16 *input_M = generate_2D_matrix(L0_h1, L0_w1, 1);
    bfloat16 *input_N = generate_2D_matrix(L0_w1, L0_w2, 1);

    std::cout << "Matrix generated!!" << std::endl;

    // print_2d_mem("Matrix M", input_M, M_size);
    // print_2d_mem("Matrix N", input_N, N_size);

    mm_graph0.in_lhs[0].gm2aie(input_M, sizeof(*input_M) * L0_h1 * L0_w1);
    mm_graph0.in_rhs[0].gm2aie(input_N, sizeof(*input_N) * L0_w1 * L0_w2);
    mm_graph0.run(1);
    std::cout << "GMIO transactions finished" << std::endl;
    mm_graph0.end();
    return 0;
}
#endif