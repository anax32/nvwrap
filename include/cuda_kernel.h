#include "cuda_result.h"

#ifndef CUDA_KERNEL_H
#define CUDA_KERNEL_H
class cuda_kernel : public cuda_result
{
public:
    typedef std::map<std::string, CUfunction>   function_list;

protected:
    CUmodule        module_;
    function_list   entry_points;

    void get_entry_points_from_module(const std::initializer_list<std::string>& entry_point_names)
    {
        std::transform(
            std::begin(entry_point_names),
            std::end(entry_point_names),
            std::inserter(entry_points, entry_points.end()),
            [&](const std::string& fn_name) -> function_list::value_type
            {
                CUfunction  fn = NULL;
                cuModuleGetFunction(&fn, module_, fn_name.c_str());
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
    template<typename T>
    void apply(const std::string& function_name, std::initializer_list<T> params)
    {
        // get the function
        auto fn = entry_points.begin()->second;

        if (function_name.length() > 0)
        {
            fn = entry_points[function_name];
        }

        // construct the void* params
        std::vector<const void*>  v_params;

        for (auto& p : params)
        {
            v_params.push_back(&p);
        }

        // run the kernel
        result_ = cuLaunchKernel(
            fn,
            1, 1, 1,
            1, 1, 1,
            0,
            NULL,
            const_cast<void**>(v_params.data()),
            NULL);

        // wait for the kernel
        result_ = cuCtxSynchronize();
    }
    template<typename T>
    void apply(std::initializer_list<T> params)
    {
        apply(entry_points.begin()->first, params);
    }
};
#endif