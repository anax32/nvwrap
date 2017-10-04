#include "cuda_kernel.h"

#ifndef CUDA_KERNEL_PTX_H
#define CUDA_KERNEL_PTX_H
// compile the kernel into a ptx file with nvcc:
//   %CUDA_PATH%\bin\nvcc --ptx -o <output>.ptx <input>.cu
class cuda_kernel_ptx : public cuda_kernel
{
public:
    cuda_kernel_ptx (std::string ptx_filename, const std::initializer_list<std::string>& entry_point_names)
    {
        result_ = cuModuleLoad (&module_, ptx_filename.c_str ());

        if (*this == true)
        {
            get_entry_points_from_module (entry_point_names);
        }
    }
    virtual ~cuda_kernel_ptx ()
    {}
};
#endif