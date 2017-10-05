// also link to:
//   nvrtc.lib
#include "cuda_kernel.h"
#include "nvrtc_program.h"

#ifndef CUDA_KERNEL_RUNTIME_H
#define CUDA_KERNEL_RUNTIME_H
class cuda_kernel_runtime : public cuda_kernel
{
public:
    cuda_kernel_runtime (const std::string& src,
                         const std::initializer_list<const char *>& entry_point_names)
    {
        nvrtc_program compilation (src);

        if (compilation == true)
        {
            auto ptx_code = compilation.get_ptx ();

            result_ = cuModuleLoadDataEx (
                &module_,
                ptx_code.data (),
                0,
                NULL,
                NULL);

            get_entry_points_from_module (entry_point_names);
        }
        else
        {
            // get log?
#ifdef _IOSTREAM_
            std::cerr << compilation.get_log().c_str() << std::endl;
#endif
        }
    }
    virtual ~cuda_kernel_runtime ()
    {}
};
#endif