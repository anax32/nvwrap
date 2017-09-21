#include "cuda_result.h"

#ifndef CUDA_INITIALISE_H
#define CUDA_INITIALISE_H
class cuda_initialise : public cuda_result
{
protected:
	CUdevice	device_;
	CUcontext	context_;
public:
	cuda_initialise()
	{
		result_ = cuInit(0);
		result_ = cuDeviceGet(&device_, 0);
		result_ = cuCtxCreate(&context_, 0, device_);
	}
	virtual ~cuda_initialise()
	{
		result_ = cuCtxDetach(context_);
	}
};
#endif