#include <map>

#ifndef CUDNN_RESULT_H
#define CUDNN_RESULT_H
class cudnn_result
{
public:
	static std::map<cudnnStatus_t, const char *>	cudnn_error_codes;
#ifdef CUBLASAPI
	static std::map<cublasStatus_t, cudnnStatus_t>	cublas_status_codes;
#endif

protected:
	cudnnStatus_t	result_;

public:
	cudnn_result()
		: result_ (CUDNN_STATUS_SUCCESS)
	{}
	void print()
	{
#ifdef _IOSTREAM_
		std::cout << "cuDNN: " << cudnn_error_codes[result_] << std::endl;
#endif
	}
	cudnnStatus_t result() const
	{
		return result_;
	}

  operator bool () const
  {
      return result () == CUDNN_STATUS_SUCCESS;
  }
};

std::map<cudnnStatus_t, const char*> cudnn_result::cudnn_error_codes = {
	{ CUDNN_STATUS_SUCCESS, "CUDNN_STATUS_SUCCESS\0" },
	{ CUDNN_STATUS_NOT_INITIALIZED, "CUDNN_STATUS_NOT_INITIALIZED\0" },
	{ CUDNN_STATUS_ALLOC_FAILED, "CUDNN_STATUS_ALLOC_FAILED\0" },
	{ CUDNN_STATUS_BAD_PARAM, "CUDNN_STATUS_BAD_PARAM\0" },
	{ CUDNN_STATUS_INTERNAL_ERROR, "CUDNN_STATUS_INTERNAL_ERROR\0" },
	{ CUDNN_STATUS_INVALID_VALUE, "CUDNN_STATUS_INVALID_VALUE\0" },
	{ CUDNN_STATUS_ARCH_MISMATCH, "CUDNN_STATUS_ARCH_MISMATCH\0" },
	{ CUDNN_STATUS_MAPPING_ERROR, "CUDNN_STATUS_MAPPING_ERROR\0" },
	{ CUDNN_STATUS_EXECUTION_FAILED, "CUDNN_STATUS_EXECUTION_FAILED\0" },
	{ CUDNN_STATUS_NOT_SUPPORTED, "CUDNN_STATUS_NOT_SUPPORTED\0" },
	{ CUDNN_STATUS_LICENSE_ERROR, "CUDNN_STATUS_LICENSE_ERROR\0" },
	{ CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING, "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING\0" },
};

#ifdef CUBLASAPI
std::map<cublasStatus_t, cudnnStatus_t> cudnn_result::cublas_status_codes = {
	{ CUBLAS_STATUS_SUCCESS, CUDNN_STATUS_SUCCESS },
	{ CUBLAS_STATUS_NOT_INITIALIZED, CUDNN_STATUS_NOT_INITIALIZED },
	{ CUBLAS_STATUS_ALLOC_FAILED, CUDNN_STATUS_ALLOC_FAILED },
	{ CUBLAS_STATUS_INVALID_VALUE, CUDNN_STATUS_INVALID_VALUE },
	{ CUBLAS_STATUS_ARCH_MISMATCH, CUDNN_STATUS_ARCH_MISMATCH },
	{ CUBLAS_STATUS_MAPPING_ERROR, CUDNN_STATUS_MAPPING_ERROR },
	{ CUBLAS_STATUS_EXECUTION_FAILED, CUDNN_STATUS_EXECUTION_FAILED },
	{ CUBLAS_STATUS_INTERNAL_ERROR, CUDNN_STATUS_INTERNAL_ERROR },
	{ CUBLAS_STATUS_NOT_SUPPORTED, CUDNN_STATUS_NOT_SUPPORTED },
	{ CUBLAS_STATUS_LICENSE_ERROR, CUDNN_STATUS_LICENSE_ERROR }
};
#endif

#endif