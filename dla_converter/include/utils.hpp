#ifndef UTILS_H
#define UTILS_H
#include <string>
#include <cuda_runtime.h>
#include <iostream>

namespace utils
{
  #define TRT_CHECK_RETURN(value) utils::CheckTRTSucc(__FILE__, __LINE__, value);
  #define CUDA_CHECK_RETURN(value) utils::CheckCudaErrorAux(__FILE__, __LINE__, #value, value);

  struct WorkSpace
  {
    std::string calib_file;
    size_t maxWorkspaceSize = 1024 * 1024 * 2;
    int log_stat = 0;
    int maxBatchSize = 16;
    int batchSize = 7; // 7 views for default
    bool wINT8 = true;  
    // false use FP16
    // DLA Only support int8 & fp16

    // NOTE: nbInps & nbOups will autoset in process
    int nbInps = -1;
    int nbOups = -1;
  };

  struct GPUTimer
  {
    cudaEvent_t start, stop;
    float elapsedTime;
    GPUTimer()
    {
      cudaEventCreate(&start);
      cudaEventRecord(start, 0);
    }
    float end()
    {
      cudaEventCreate(&stop);
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&elapsedTime, start, stop);
      return elapsedTime; // ms
    }
    float end(std::string str)
    {
      std::cout << str;
      return end();
    }
  };


  void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err);

  void CheckTRTSucc(const char* file, unsigned line, bool succ);

  // get filename, eg. a/c/dd.txt -> dd
  std::string getFilename(std::string filename);

  // check file exits
  bool fileExist(const std::string &filepath);

  

} // namespace utils

#endif // UTILS_H