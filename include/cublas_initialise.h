#include "cublas_result.h"

#ifndef CUBLAS_INITIALISE_H
#define CUBLAS_INITIALISE_H
class cublas_initialise : public cublas_result
{
protected:
	cublasHandle_t	cublas_context_;
public:
	cublas_initialise()
		: cublas_context_(NULL)
	{
		result_ = cublasCreate(&cublas_context_);
	}
	virtual ~cublas_initialise()
	{
		result_ = cublasDestroy(cublas_context_);
	}
	operator cublasHandle_t() const	{ return cublas_context_; }
};
#endif