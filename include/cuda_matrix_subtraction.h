#include "cuda_kernel_runtime.h"
#include "cudnn_tensor.h"

#ifndef CUDNN_MATRIX_SUBTRACTION_H
#define CUDNN_MATRIX_SUBTRACTION_H
template<typename T>
class cuda_matrix_subtraction : public cuda_kernel_runtime
{
protected:
    // NB: the base class is initialized before the members,
    // therefore, we cannot set the source string contents here.
    // But, using a static variable allows us to use template
    // specialisation for the parameters of the kernel, so it
    // sort of has a silver lining.
    static const char*  source;

public:
    cuda_matrix_subtraction (cudnnHandle_t cudnn_context)
        : cuda_kernel_runtime(source, { {"subtract"} })
    {}
    template<typename T>
    void apply(const cudnn_tensor<T>& A,
               const cudnn_tensor<T>& B,
               cudnn_tensor<T>& C)
    {
        cuda_kernel_runtime::apply(
            {
                A.device_storage(),
                B.device_storage(),
                C.device_storage(),
                (void*)A.count()
            },
            {static_cast<unsigned int>(A.count())});
    }
};

// single precision matrix subtraction
const char * cuda_matrix_subtraction<float>::source = 
"                                      \n\
extern \"C\" __global__ void subtract (float *A, float *B, float *C, int N)    \n\
{                                                               \n\
    int i = blockDim.x * blockIdx.x + threadIdx.x;              \n\
                                                                \n\
    if (i < N)                                                  \n\
    {                                                           \n\
        C[i] = A[i] - B[i];                                     \n\
    }                                                           \n\
}                                                               \n";

// double precision matrix subtraction
const char * cuda_matrix_subtraction<double>::source =
"                                      \n\
extern \"C\" __global__ void subtract (double *A, double *B, double *C, int N)    \n\
{                                                               \n\
    int i = blockDim.x * blockIdx.x + threadIdx.x;              \n\
                                                                \n\
    if (i < N)                                                  \n\
    {                                                           \n\
        C[i] = A[i] - B[i];                                     \n\
    }                                                           \n\
}                                                               \n";

#endif