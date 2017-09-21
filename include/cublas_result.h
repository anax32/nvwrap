#ifndef CUBLAS_RESULT_H
#define CUBLAS_RESULT_H
class cublas_result
{
protected:
	cublasStatus_t	result_;
public:
	cublas_result()
		: result_(CUBLAS_STATUS_SUCCESS)
	{}
	cublasStatus_t result() const
	{
		return result_;
	}
};
#endif