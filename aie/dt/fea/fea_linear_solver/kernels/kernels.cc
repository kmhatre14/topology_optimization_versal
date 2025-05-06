#include "aie_api/vector.hpp"
#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>
// #include <iostream>
#include <stdio.h>

const unsigned int nelx = 6;
const unsigned int nely = 2;
// const bfloat16 penal = 3.0;

void FE_kernel(adf::input_buffer<float> &x_in, adf::output_buffer<float> &U_out,
               adf::output_buffer<float> &K_out, adf::output_buffer<float> &F_out,
               adf::output_buffer<float> &KE_out) {

    unsigned long long cycle_num;
    aie::tile tile = aie::tile::current();
    cycle_num = tile.cycles();
    printf("Start Cycle=%llu\n", cycle_num);

    const float *x = x_in.data();
    constexpr float KE[8][8] = {{0.494505, 0.178571, -0.302198, -0.013736, -0.247253,
                                 -0.178571, 0.054945, 0.013736},
                                {0.178571, 0.494505, 0.013736, 0.054945, -0.178571,
                                 -0.247253, -0.013736, -0.302198},
                                {-0.302198, 0.013736, 0.494505, -0.178571, 0.054945,
                                 -0.013736, -0.247253, 0.178571},
                                {-0.013736, 0.054945, -0.178571, 0.494505, 0.013736,
                                 -0.302198, 0.178571, -0.247253},
                                {-0.247253, -0.178571, 0.054945, 0.013736, 0.494505,
                                 0.178571, -0.302198, -0.013736},
                                {-0.178571, -0.247253, -0.013736, -0.302198, 0.178571,
                                 0.494505, 0.013736, 0.054945},
                                {0.054945, -0.013736, -0.247253, 0.178571, -0.302198,
                                 0.013736, 0.494505, -0.178571},
                                {0.013736, -0.302198, 0.178571, -0.247253, -0.013736,
                                 0.054945, -0.178571, 0.494505}};
    constexpr int ndof = 2 * (nelx + 1) * (nely + 1);
    float U[ndof] = {0};
    float F[ndof] = {0};
    float K[ndof][ndof] = {0};
    for (int elx = 0; elx < nelx; ++elx) {
        for (int ely = 0; ely < nely; ++ely) {
            int n1 = (nely + 1) * elx + ely;
            int n2 = (nely + 1) * (elx + 1) + ely;
            int edof[8] = {2 * n1,     2 * n1 + 1, 2 * n2,     2 * n2 + 1,
                           2 * n2 + 2, 2 * n2 + 3, 2 * n1 + 2, 2 * n1 + 3};
            float coeff = 1.0;
            for (int i = 0; i < 3.0; ++i) {
                coeff *= x[elx + ely * (nelx)];
            }
            for (int i = 0; i < 8; ++i) {
                aie::vector<float, 8> KE_row;
                aie::vector<float, 8> mul_result;
                aie::vector<float, 8> vec;
                aie::vector<float, 8> K_new;
                float temp[8];
                for (int k = 0; k < 8; ++k) {
                    temp[k] = K[edof[i]][edof[k]];
                }
                vec = aie::load_v<8>(temp);
                mul_result = aie::mul(KE_row, coeff);
                K_new = aie::add(vec, mul_result);
                float K_new_val[8];
                aie::store_v(K_new_val, K_new);
                for (int j = 0; j < 8; ++j) {
                    K[edof[i]][edof[j]] = K_new_val[j];
                }
            }
        }
    }

    // Print cycles
    cycle_num = tile.cycles();
    printf("After K gen Cycle=%llu\n", cycle_num);

    F[1] = -1.0f;
    int count = 0;
    int fixeddofs[2 * (nely + 1) + 1];
    for (int i = 0; i <= nely; ++i) {
        fixeddofs[count++] = 2 * i;
    }
    fixeddofs[count++] = ndof - 1;
    int alldofs[ndof];
    for (int i = 0; i < ndof; ++i) {
        alldofs[i] = i;
    }
    int freedofs[ndof];
    for (int i = 0; i < count - 1; ++i) {
        for (int j = i + 1; j < count; ++j) {
            if (fixeddofs[j] < fixeddofs[i]) {
                int temp = fixeddofs[i];
                fixeddofs[i] = fixeddofs[j];
                fixeddofs[j] = temp;
            }
        }
    }
    int fix_index = 0, all_index = 0, counter = 0;
    while (all_index < ndof) {
        if (fix_index < count && alldofs[all_index] == fixeddofs[fix_index]) {
            ++fix_index;
        } else {
            freedofs[counter++] = alldofs[all_index];
        }
        ++all_index;
    }
    int nfree = counter;
    float K_reduced[ndof][ndof];
    float F_reduced[ndof];
    for (int i = 0; i < nfree; ++i) {
        for (int j = 0; j < nfree; ++j) {
            K_reduced[i][j] = K[freedofs[i]][freedofs[j]];
        }
        F_reduced[i] = F[freedofs[i]];
    }

    alignas(32) float mat[ndof][ndof];
    alignas(8) float mat_copy[ndof * ndof];
    alignas(8) float rhs[ndof];
    alignas(32) float z[ndof];
    for (int i = 0; i < nfree; ++i) {
        for (int j = 0; j < nfree; ++j) {
            mat[i][j] = K_reduced[i][j];
        }
        rhs[i] = F_reduced[i];
        z[i] = 0.0f;
    }
    int i = 0;
    for (; i + 8 <= nfree; i += 8) {
        int pivot = i;
        for (int j = i + 1; j < nfree; ++j) {
            float val1 = mat[j][i];
            if (val1 < 0)
                val1 = -val1;
            float val2 = mat[pivot][i];
            if (val2 < 0)
                val2 = -val2;
            if (val1 > val2) {
                pivot = j;
            }
        }
        if (pivot != i) {
            for (int k = 0; k < nfree; ++k) {
                float temp = mat[i][k];
                mat[i][k] = mat[pivot][k];
                mat[pivot][k] = temp;
            }
            float tempRhs = rhs[i];
            rhs[i] = rhs[pivot];
            rhs[pivot] = tempRhs;
        }
        z[i] = rhs[i];
        int j = i + 1, k = i;
        float factor = 0;
        aie::vector<float, 8> matVec;
        aie::vector<float, 8> zVec;
        aie::vector<float, 8> row_j;
        aie::vector<float, 8> row_i;
        for (; j + 8 <= nfree; j += 8) {
            factor = mat[j][i] / mat[i][i];
            for (; k < nfree; ++k) {
                mat[j][k] -= factor * mat[i][k];
            }
            zVec = aie::load_v<8>(&z[i]);
            auto prod = aie::mul(factor, zVec);
            auto vec = prod.to_vector();
            auto sum = aie::reduce_add_v(vec);
            zVec = aie::sub(zVec, sum);
            // aie::store_v(&z[i], zVec);
            count++;
        }
        for (; j < nfree; ++j) {
            float factor = mat[j][i] / mat[i][i];
            z[j] -= factor * z[i];
        }
    }
    // rhs not working in vectorization so clearing out z prior to back substitution
    for (int l = 0; l < nfree; ++l) {
        rhs[i] = z[i];
        z[i] = 0.0f;
    }
    // for (int j = 0; j < 42; ++j) {
    //     mat[0][j] = 12.34f;
    // }
    // Print cycles
    cycle_num = tile.cycles();
    printf("After elimination Cycle=%llu\n", cycle_num);

    for (; i >= 7; i -= 8) {
        z[i] = rhs[i];
        int j = i + 1;
        aie::vector<float, 8> matVec;
        aie::vector<float, 8> zVec;
        for (; j + 8 <= nfree; j += 8) {
            // printf("j=%d and i =%d\n", j, i);

            // aie::load_v cannot work with unaligned data, thus performing a scalar load
            for (int k = 0; k < 8; ++k) {
                matVec[k] = mat[i][j + k];
            }
            // Avoid using aie::load_v for unaligned data
            // matVec = aie::load_v<8>(&mat[i][j]);

            // aie::print(matVec, true, "matVec=");

            // Avoid using aie::load_v for unaligned data
            // j-index: 25,17,25,9,17,25
            for (int k = 0; k < 8; ++k) {
                zVec[k] = z[j + k];
            }
            // zVec = aie::load_v<8>(&z[j]);

            auto prod = aie::mul(matVec, zVec);
            auto vec = prod.to_vector();
            auto sum = aie::reduce_add_v(vec);
            zVec = aie::sub(zVec, sum);

            // Avoid using aie::store_v for unaligned data
            // j index: 25,17,25,9,17,25
            for (int k = 0; k < 8; ++k) {
                z[j + k] = zVec[k];
            }
            // aie::store_v(&z[j], zVec);
        }
        for (; j < nfree; ++j) {
            z[i] -= mat[i][j] * z[j];
        }
        z[i] /= mat[i][i];
    }

    // Print cycles
    cycle_num = tile.cycles();
    printf("After Back substitution Cycle=%llu\n", cycle_num);

    for (int i = 0; i < nfree; ++i) {
        U[freedofs[i]] = z[i];
    }
    int fixeddofs_len = 2 * (nely + 1) + 1;
    for (int i = 0; i < fixeddofs_len; ++i) {
        U[fixeddofs[i]] = 0.0;
    }
    float *U_ptr = U_out.data();
    float *F_ptr = F_out.data();
    float *K_ptr = K_out.data();
    float *KE_ptr = KE_out.data();
    for (int i = 0; i < ndof; ++i) {
        U_ptr[i] = U[i];
    }
    for (int i = 0; i < ndof; ++i) {
        F_ptr[i] = F[i];
    }
    int index = 0;
    for (int i = 0; i < ndof; ++i) {
        for (int j = 0; j < ndof; ++j) {
            K_ptr[index++] = K[i][j];
        }
    }
    index = 0;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            KE_ptr[index++] = KE[i][j];
        }
    }
    // printf("%s\n", "U first 10 elements:");
    // for (int i = 0; i < 10 && i < ndof; ++i) {
    //     printf("%f\n", U[i]);
    // }
    // printf("%s\n", "\n");
    // printf("%s\n", "F first element:");
    // for (int i = 0; i < 20 && i < ndof; ++i) {
    //     printf("%f\n", U[i]);
    // }
    // printf("%s\n", "\n");
    // printf("%s\n", "K first 10 elements:");
    // for (int i = 0; i < 64; ++i) {
    //     for (int j = 0; j < 8; ++j) {
    //         printf("%f\n", K[i][j]);
    //     }
    // }
}