#ifndef TRT_H
#define TRT_H
#include <string>

#include <cstdlib>
#include <unordered_map>
#include "cuda_runtime_api.h"
#include "NvOnnxParser.h"
#include "cuda.h"
#include "cuda_fp16.h"
#include <iostream>
#include "utils.hpp"
#include <memory>


namespace trt
{
  using namespace utils;
  static auto StreamDeleter = [](cudaStream_t *pStream)
  {
    if (pStream)
    {
      cudaStreamDestroy(*pStream);
      delete pStream;
    }
  };

  inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream()
  {
    std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(new cudaStream_t, StreamDeleter);
    if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess)
    {
      pStream.reset(nullptr);
    }
    return pStream;
  }  

  

  // convert calibation_file to map 
  std::unordered_map<std::string, float> getCalib(std::string files);
  void printCalib(std::unordered_map<std::string, float> map);

  // logger for trt
  class Logger : public nvinfer1::ILogger
  {
    public:
      Logger(){};
      Logger(int stat) { stat_ = stat; };
        void log(Severity severity, const char *msg) noexcept override
        {
          if (stat_ == 0) // print all
          {
            std::cout << msg << std::endl;
          }
          else if (stat_ == 1 && (severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) 
          {
            std::cout << msg << std::endl;
          }
        }
    private:
      int stat_ = 0;
  };

  /**
   * @brief TensorRT Engine builder. Used utils::WorkSpace to configure.
   * 
   */
  class EngineBuilder {
    public:
        EngineBuilder(const utils::WorkSpace &work_space): wSpace(work_space) {
        logger = Logger(wSpace.log_stat);
        if (wSpace.batchSize > wSpace.maxBatchSize) {
          std::cout << wSpace.batchSize  << ">" << wSpace.maxBatchSize << std::endl;
          throw std::runtime_error("Batchsize is larger than max batch");
        }
      };
      // fp16 dont need calib
      // user should set calib in WSpace 
      bool build(const char* onnx); 
      utils::WorkSpace wSpace;
      Logger logger;
      std::string engineName;
      
  };
    
} // namespace trt



#endif // TRT_H