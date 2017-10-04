#ifndef CUDA_RESULT_H
#define CUDA_RESULT_H
class cuda_result
{
protected:
	CUresult	result_;
public:
	cuda_result()
		: result_(CUDA_SUCCESS)
	{}
	CUresult result() const
	{
		return result_;
	}
  operator bool () const
  {
      return result () == CUDA_SUCCESS;
  }
};
#endif