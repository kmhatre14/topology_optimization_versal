
#include "mat.hpp"

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
    #ifndef AMD_VIRSAL
    this->data.arr_2d = (bfloat16 *)malloc(row * col * sizeof(bfloat16));
    #elif
    this->data.arr_2d = (bfloat16 *)GMIO::malloc(row * col * sizeof(bfloat16));
    #endif
}

mat::~mat()
{
    this->data.arr_2d = nullptr;
    this->data.rows = 0;
    this->data.cols = 0;
    this->data.size = 0;
}

FUNC_RET mat::mat_alloc(int rows, int cols)
{
    if (this->data.arr_2d == nullptr)
    {
        #ifndef AMD_VIRSAL
        this->data.arr_2d = (bfloat16 *)malloc(rows * cols * sizeof(bfloat16));
        #elif
        this->data.arr_2d = (bfloat16 *)GMIO::malloc(rows * cols * sizeof(bfloat16));
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
        #ifndef AMD_VIRSAL
        this->data.arr_2d = (bfloat16 *)malloc(this->data.rows * this->data.cols * sizeof(bfloat16));
        #elif
        this->data.arr_2d = (bfloat16 *)GMIO::malloc(this->data.rows * this->data.cols * sizeof(bfloat16));
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

FUNC_RET mat::mat_mul(mat N, mat*const out)
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
            sum = 0;
            for(int k = 0; k< this->data.cols; k++)
            {
                sum += this->mat_at(i,k) * N.mat_at(k,j);
            }
            out->mat_at(i,j) = sum;
        }
    }
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
        return FUNC_NOK;
    }

    for (int i = 0; i < this->data.rows; ++i) {
        for (int j = 0; j < this->data.cols; ++j) {
            this->mat_at(i,j) = tensor[i][j];
        }
    }
    return FUNC_OK;
}
