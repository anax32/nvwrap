// also link to:
//   nvrtc.lib

#include <nvrtc.h>

#ifndef NVRTC_RESULT_H
#define NVRTC_RESULT_H
class nvrtc_result
{
protected:
    nvrtcResult result_;
public:
    nvrtc_result()
        : result_(NVRTC_SUCCESS)
    {}
    nvrtcResult result() const
    {
        return result_;
    }
    operator bool() const
    {
        return result() == NVRTC_SUCCESS;
    }
};
#endif