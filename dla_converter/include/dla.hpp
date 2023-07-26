#ifndef DLA_H
#define DLA_H
#include "cudla.h"
#include "trt.hpp"
#include <vector>
#include <iostream>
#include "NvOnnxParser.h"


#define DPRINTF(...) printf(__VA_ARGS__)
#define NVDLA_CHECK_RETURN(err)                        \
if (err != cudlaSuccess) {                             \
  std::cout << "error code is " << err << std::endl;   \
  exit(-1);                                            \
};

namespace dla {
  static void printTensorDesc(cudlaModuleTensorDescriptor* tensorDesc);
  typedef struct {
    cudlaDevHandle dla_handler;
    cudlaModule module_handler;
    std::vector<uint64_t*> oup_regptrs;
    std::vector<uint64_t*> inp_regptrs;
    cudlaModuleTensorDescriptor* inputTensorDesc;
    cudlaModuleTensorDescriptor* outputTensorDesc;
  } DLAResource;
  // clean dla resource
  void cleanDLAResource(DLAResource* resource);
  bool cudlaLoadTRT(const char* trt, DLAResource &resource);

  template <typename dtype>
  dtype *makeCPUDummpyInputs(int32_t size)
  {
    dtype *inp = (dtype *)malloc(size * sizeof(dtype));
    for (int i = 0; i < size; ++i)
    {
      inp[i] = static_cast<dtype>((float(i) / 2.) / 100.) + static_cast<dtype>(0.5);
    }
    return inp;
  }


  class DLAConverter {
    public:
      bool verbose = true;
      DLAConverter(const char* onnx, utils::WorkSpace wSpace);
      ~DLAConverter();
      bool cudla_forward(int numInps, int numOups);
    private:
      DLAResource resource_;
      std::vector<void *> predicitonBindings{2};
  };


} // dla


#endif // DLA_H