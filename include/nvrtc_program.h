// also link to:
//   nvrtc.lib
#include <nvrtc.h>

#include "nvrtc_result.h"

#include <vector>

#ifndef NVRTC_PROGRAM_H
#define NVRTC_PROGRAM_H
class nvrtc_program : public nvrtc_result
{
public:
    typedef std::vector<char>   ptx_code;

protected:
    ptx_code        ptx_code_;
    nvrtcProgram    program_;

public:
    nvrtc_program (std::string source, std::vector<const char*> options = {})
    {
        result_ = nvrtcCreateProgram (
            &program_,
            source.c_str (),
            NULL,
            0,
            NULL,
            NULL);

        result_ = nvrtcCompileProgram (
            program_,
            static_cast<int>(options.size ()),
            options.data ());
    }
    virtual ~nvrtc_program ()
    {
        result_ = nvrtcDestroyProgram (&program_);
    }
    std::string get_log ()
    {
        std::string log;
        size_t			log_len;

        result_ = nvrtcGetProgramLogSize (
            program_,
            &log_len);

        log.resize (log_len + 1);

        result_ = nvrtcGetProgramLog (
            program_,
            const_cast<char*>(log.data ()));

        return log;
    }
    ptx_code get_ptx ()
    {
        // get compiled code
        size_t			ptx_len = 0;
        ptx_code    ptx_src;

        result_ = nvrtcGetPTXSize (program_, &ptx_len);
        ptx_src.resize (ptx_len);
        result_ = nvrtcGetPTX (program_, ptx_src.data ());

        return ptx_src;
    }
};
#endif