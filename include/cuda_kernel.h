#include "cuda_result.h"

#include <vector>
#include <map>
#include <algorithm>
#include <iterator>

#ifndef CUDA_KERNEL_H
#define CUDA_KERNEL_H
class cuda_kernel : public cuda_result
{
public:
    typedef std::map<std::string, CUfunction>   function_list;

protected:
    CUmodule        module_;
    function_list   entry_points;

    void get_entry_points_from_module(const std::initializer_list<const char *>& entry_point_names)
    {
        std::transform(
            std::begin(entry_point_names),
            std::end(entry_point_names),
            std::inserter(entry_points, entry_points.end()),
            [&](const char* fn_name) -> function_list::value_type
            {
                CUfunction  fn = NULL;
                cuModuleGetFunction(&fn, module_, fn_name);
                return std::make_pair(fn_name, fn);
            });
    }
public:
    cuda_kernel ()
        : module_ (NULL)
    {}
    virtual ~cuda_kernel()
    {
        cuModuleUnload (module_);
        module_ = NULL;
    }
    // execution a kernel by calling function_name
    // and passing params to each invocation.
    // The number of invocations is given by a vector,
    // this should correspond to the number of threads-per-block
    // FIXME: currently only invocation[0] is read, and there
    // is no communication of block_size from/to the caller.
    template<typename T>
    void apply(const std::string& function_name,
               std::initializer_list<T> params,
               const std::vector<unsigned int>& invocations)
    {
        // get the function
        auto fn = entry_points.begin()->second;

        if (function_name.length() > 0)
        {
            fn = entry_points[function_name];
        }

        // construct the void* params
        std::vector<const void*>  kernel_params;

        for (auto& p : params)
        {
            kernel_params.push_back(&p);
        }

        // estimate the block dimensions
        // FIXME: its likely this will need to be worked over,
        // ultimately, the caller should have control of this
        // information.
        const size_t block_size = 512;
        auto nblocks = std::max (invocations[0] / block_size, size_t(1));

        // run the kernel
        result_ = cuLaunchKernel(
            fn,
            static_cast<unsigned int>(nblocks), 1, 1,
            static_cast<unsigned int>(block_size), 1, 1,
            0,
            NULL,
            const_cast<void**>(kernel_params.data()),
            NULL);

        // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15
        switch (result_)
        {
        case CUDA_SUCCESS:
            break;
        case CUDA_ERROR_INVALID_VALUE:
            // both kernelParams (v_params) and extra parameters have
            // have been passed to cuLaunchKernel
            break;
        case CUDA_ERROR_INVALID_IMAGE:
            // the compiled kernel does not have parameter information
            // to allow v_params to be mapped correctly.
            break;
        }

        // wait for the kernel
        result_ = cuCtxSynchronize();
    }
    template<typename T>
    void apply(std::initializer_list<T> params, 
               const std::vector<unsigned int>& invocations)
    {
        apply(
            entry_points.begin()->first,
            params,
            invocations);
    }
};
#endif