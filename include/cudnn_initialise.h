#include "cudnn_result.h"

#ifndef CUDNN_INITIALISE_H
#define CUDNN_INITIALISE_H
class cudnn_initialise : public cudnn_result
{
protected:
	cudnnHandle_t	handle_;

public:
	cudnn_initialise()
	{
		result_ = cudnnCreate(&handle_);
	}
	virtual ~cudnn_initialise()
	{
		result_ = cudnnDestroy(handle_);
	}
	operator cudnnHandle_t () const { return handle_; }
};
#endif