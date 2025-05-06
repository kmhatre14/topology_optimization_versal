#include <adf.h>
#include <cstdio>
#include <stdio.h>
// #include "adf/io_buffer/io_buffer_types.h"
#include "aie_api/detail/config.hpp"
#include "aie_api/types.hpp"
#include <aie_api/aie.hpp>
// #include "include.h"
#include "optional"

const unsigned int nelx = 6;
const unsigned int nely = 2;
// const bfloat16 penal = 3.0;

void FE_kernel(adf::input_buffer<float> &x_in, adf::output_buffer<float> &U_out,
               adf::output_buffer<float> &K_out, adf::output_buffer<float> &F_out,
               adf::output_buffer<float> &KE_out) {

    printf("Start of the Kernel\n");
    unsigned long long cycle_num;
    aie::tile tile = aie::tile::current();
    cycle_num = tile.cycles();
    printf("Cycle=%llu\n", cycle_num);
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
    printf("Line 37\n");
    cycle_num = tile.cycles();
    printf("Cycle=%llu\n", cycle_num);
    constexpr int ndof = 2 * (nelx + 1) * (nely + 1);
    float U[ndof] = {0};
    float F[ndof] = {0};
    float K[ndof][ndof] = {0};
    printf("Line 42\n");
    cycle_num = tile.cycles();
    printf("Cycle=%llu\n", cycle_num);
    for (int elx = 0; elx < nelx; ++elx) {
        for (int ely = 0; ely < nely; ++ely) {
            printf("Line 45\n");
            int n1 = (nely + 1) * elx + ely;
            int n2 = (nely + 1) * (elx + 1) + ely;
            int edof[8] = {2 * n1,     2 * n1 + 1, 2 * n2,     2 * n2 + 1,
                           2 * n2 + 2, 2 * n2 + 3, 2 * n1 + 2, 2 * n1 + 3};
            float coeff = 1.0;
            printf("Line 49\n");
            for (int i = 0; i < 3.0; ++i) {
                coeff *= x[elx + ely * (nelx)];
            }
            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; ++j) {
                    K[edof[i]][edof[j]] += coeff * KE[i][j];
                }
            }
        }
    }
    cycle_num = tile.cycles();
    printf("Cycle=%llu\n", cycle_num);
    printf("Line 58\n");
    F[1] = -1.0f;
    int fixeddofs[2 * (nely + 1) + 1];
    int count = 0;
    for (int i = 0; i <= nely; ++i) {
        fixeddofs[count++] = 2 * i;
    }
    cycle_num = tile.cycles();
    printf("76 Cycle=%llu\n", cycle_num);
    fixeddofs[count++] = ndof - 1;
    int alldofs[ndof];
    for (int i = 0; i < ndof; ++i) {
        alldofs[i] = i;
    }
    cycle_num = tile.cycles();
    printf("82 Cycle=%llu\n", cycle_num);
    printf("Line 70\n");
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
    cycle_num = tile.cycles();
    printf("94 Cycle=%llu\n", cycle_num);
    printf("Line 81\n");
    int fix_index = 0, all_index = 0, counter = 0;
    while (all_index < ndof) {
        if (fix_index < count && alldofs[all_index] == fixeddofs[fix_index]) {
            ++fix_index;
        } else {
            freedofs[counter++] = alldofs[all_index];
        }
        ++all_index;
    }
    cycle_num = tile.cycles();
    printf("105 Cycle=%llu\n", cycle_num);
    int nfree = counter;
    float K_reduced[ndof][ndof];
    float F_reduced[ndof];
    for (int i = 0; i < nfree; ++i) {
        for (int j = 0; j < nfree; ++j) {
            K_reduced[i][j] = K[freedofs[i]][freedofs[j]];
        }
        F_reduced[i] = F[freedofs[i]];
    }
    cycle_num = tile.cycles();
    printf("115 Cycle=%llu\n", cycle_num);
    printf("Line 100\n");
    float mat[ndof][ndof];
    float rhs[ndof];
    float z[ndof];
    for (int i = 0; i < nfree; ++i) {
        for (int j = 0; j < nfree; ++j) {
            mat[i][j] = K_reduced[i][j];
        }
        rhs[i] = F_reduced[i];
        z[i] = 0.0f;
    }
    cycle_num = tile.cycles();
    printf("127 Cycle=%llu\n", cycle_num);
    printf("Line 111\n");
    for (int i = 0; i < nfree; ++i) {
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
        cycle_num = tile.cycles();
        printf("115 Cycle=%llu\n", cycle_num);
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
        cycle_num = tile.cycles();
        printf("164 Cycle=%llu\n", cycle_num);
        for (int j = i + 1; j < nfree; ++j) {
            float factor = mat[j][i] / mat[i][i];
            for (int k = i; k < nfree; ++k) {
                mat[j][k] -= factor * mat[i][k];
            }
            rhs[j] -= factor * rhs[i];
        }
        cycle_num = tile.cycles();
        printf("173 Cycle=%llu\n", cycle_num);
    }
    printf("Line 143\n");
    for (int i = nfree - 1; i >= 0; --i) {
        z[i] = rhs[i];
        for (int j = i + 1; j < nfree; ++j) {
            z[i] -= mat[i][j] * z[j];
        }
        z[i] /= mat[i][i];
    }
    cycle_num = tile.cycles();
    printf("184 Cycle=%llu\n", cycle_num);
    printf("Line 153\n");
    for (int i = 0; i < nfree; ++i) {
        U[freedofs[i]] = z[i];
    }
    cycle_num = tile.cycles();
    printf("190 Cycle=%llu\n", cycle_num);
    printf("Line 157\n");
    int fixeddofs_len = 2 * (nely + 1) + 1;
    for (int i = 0; i < fixeddofs_len; ++i) {
        U[fixeddofs[i]] = 0.0;
    }
    cycle_num = tile.cycles();
    printf("197 Cycle=%llu\n", cycle_num);
    printf("Line 162\n");
    float *U_ptr = U_out.data();
    float *F_ptr = F_out.data();
    float *K_ptr = K_out.data();
    float *KE_ptr = KE_out.data();
    for (int i = 0; i < ndof; ++i) {
        U_ptr[i] = U[i];
    }
    cycle_num = tile.cycles();
    printf("207 Cycle=%llu\n", cycle_num);
    printf("Line 170\n");
    for (int i = 0; i < ndof; ++i) {
        F_ptr[i] = F[i];
    }
    cycle_num = tile.cycles();
    printf("213 Cycle=%llu\n", cycle_num);
    printf("Line 168\n");
    int index = 0;
    for (int i = 0; i < ndof; ++i) {
        for (int j = 0; j < ndof; ++j) {
            K_ptr[index++] = K[i][j];
        }
    }
    cycle_num = tile.cycles();
    printf("222 Cycle=%llu\n", cycle_num);
    index = 0;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            KE_ptr[index++] = KE[i][j];
        }
    }
    cycle_num = tile.cycles();
    printf("213 Cycle=%llu\n", cycle_num);
}