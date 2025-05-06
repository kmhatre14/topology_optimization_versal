
#include <adf.h>
#include "adf/new_frontend/adf.h"
#include "adf/new_frontend/types.h"
// #include "kernels.h"
#include "kernels/kernel.h"

// using namespace adf;

class simpleGraph : public adf::graph {
private:
    adf::kernel fe_kernel;

public:
    adf::input_plio x_plio;
    adf::output_plio U_plio;
    adf::output_plio K_plio;
    adf::output_plio F_plio;
    adf::output_plio KE_plio;
    simpleGraph() {
        x_plio = adf::input_plio::create(adf::plio_64_bits, "x.txt");
        U_plio = adf::output_plio::create(adf::plio_64_bits, "data/U.txt");
        K_plio = adf::output_plio::create(adf::plio_64_bits, "data/K.txt");
        F_plio = adf::output_plio::create(adf::plio_64_bits, "data/F.txt");
        KE_plio = adf::output_plio::create(adf::plio_64_bits, "data/KE.txt");

        fe_kernel = adf::kernel::create(FE_kernel);

        adf::connect<>(x_plio.out[0], fe_kernel.in[0]);
        adf::connect<>(fe_kernel.out[0], U_plio.in[0]);
        adf::connect<>(fe_kernel.out[1], K_plio.in[0]);
        adf::connect<>(fe_kernel.out[2], F_plio.in[0]);
        adf::connect<>(fe_kernel.out[3], KE_plio.in[0]);

        constexpr int nelx = 6;
        constexpr int nely = 2;
        constexpr int ndof = 2 * (nelx + 1) * (nely + 1);
        constexpr int x_size = 12;

        adf::dimensions(fe_kernel.in[0]) = {x_size};
        adf::dimensions(fe_kernel.out[0]) = {ndof};
        adf::dimensions(fe_kernel.out[1]) = {ndof * ndof};
        adf::dimensions(fe_kernel.out[2]) = {ndof};
        adf::dimensions(fe_kernel.out[3]) = {64};

        adf::source(fe_kernel) = "kernels/kernels.cc";

        adf::runtime<adf::ratio>(fe_kernel) = 0.9;
    }
};
