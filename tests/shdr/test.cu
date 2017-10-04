extern "C" __global__ void test_fn_ptx (int *a, int *b)
{
	*a = 11;
    *b = 21;
}