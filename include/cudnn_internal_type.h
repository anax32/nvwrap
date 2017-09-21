#ifndef CUDNN_INTERNAL_TYPE_H
#define CUDNN_INTERNAL_TYPE_H
template<typename T>
class cudnn_internal_type
{
public:
	template<typename Q = T>
	typename std::enable_if<std::is_same<Q, float>::value, cudnnDataType_t>::type internal_type() const
	{
		return CUDNN_DATA_FLOAT;
	}

	template<typename Q = T>
	typename std::enable_if<std::is_same<Q, double>::value, cudnnDataType_t>::type internal_type() const
	{
		return CUDNN_DATA_DOUBLE;
	}

	template<typename Q = T>
	typename std::enable_if<std::is_same<Q, char>::value, cudnnDataType_t>::type internal_type() const
	{
		return CUDNN_DATA_INT8;
	}

	template<typename Q = T>
	typename std::enable_if<std::is_same<Q, int>::value, cudnnDataType_t>::type internal_type() const
	{
		return CUDNN_DATA_INT32;
	}
};
#endif