#include "dla.hpp"
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include "cudla.h"

namespace dla {
  static void printTensorDesc(cudlaModuleTensorDescriptor* tensorDesc) {
    DPRINTF("----------------------\n");
    DPRINTF("\tTENSOR NAME : %s\n", tensorDesc->name);
    DPRINTF("\tsize: %lu\n", tensorDesc->size);

    DPRINTF("\tdims: [%lu, %lu, %lu, %lu]\n", tensorDesc->n, tensorDesc->c,
            tensorDesc->h, tensorDesc->w);

    DPRINTF("\tdata fmt: %d\n", tensorDesc->dataFormat);
    DPRINTF("\tdata type: %d\n", tensorDesc->dataType);
    DPRINTF("\tdata category: %d\n", tensorDesc->dataCategory);
    DPRINTF("\tpixel fmt: %d\n", tensorDesc->pixelFormat);
    DPRINTF("\tpixel mapping: %d\n", tensorDesc->pixelMapping);
    DPRINTF("\tstride[0]: %d\n", tensorDesc->stride[0]);
    DPRINTF("\tstride[1]: %d\n", tensorDesc->stride[1]);
    DPRINTF("\tstride[2]: %d\n", tensorDesc->stride[2]);
    DPRINTF("\tstride[3]: %d\n", tensorDesc->stride[3]);
    DPRINTF("\n");
  }

  void cleanDLAResource(DLAResource *resource)
  {
    if (resource->inputTensorDesc != NULL)
    {
      free(resource->inputTensorDesc);
      resource->inputTensorDesc = NULL;
    }
    if (resource->outputTensorDesc != NULL)
    {
      free(resource->outputTensorDesc);
      resource->outputTensorDesc = NULL;
    }
    if (resource->module_handler != NULL)
    {
      NVDLA_CHECK_RETURN(cudlaModuleUnload(resource->module_handler, 0));
      resource->module_handler = NULL;
    }
    if (resource->dla_handler != NULL)
    {
      NVDLA_CHECK_RETURN(cudlaDestroyDevice(resource->dla_handler));
      resource->dla_handler = NULL;
    }
  };

  bool cudlaLoadTRT(const char* trt, DLAResource &resource) {
    cudlaModuleAttribute attribute;
    cudlaDevHandle dla_handler;
    cudlaModule module_handler;
    std::ifstream file(trt, std::ios::binary);
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(file), {});
    // default use 0
    CUDA_CHECK_RETURN(cudaSetDevice(0));
    NVDLA_CHECK_RETURN(cudlaCreateDevice(0, &dla_handler, CUDLA_CUDA_DLA)); // decice can only be 0/1
    resource.dla_handler = dla_handler;
    NVDLA_CHECK_RETURN(cudlaModuleLoadFromMemory(dla_handler, buffer.data(), buffer.size(), &module_handler, 0)); // last is flag must be 0
    resource.module_handler = module_handler;

    // request cpu memory
    NVDLA_CHECK_RETURN(cudlaModuleGetAttributes(module_handler, CUDLA_NUM_INPUT_TENSORS, &attribute));
    uint32_t numInputTensors = attribute.numInputTensors;
    NVDLA_CHECK_RETURN(cudlaModuleGetAttributes(module_handler, CUDLA_NUM_OUTPUT_TENSORS, &attribute));
    uint32_t numOutputTensors = attribute.numOutputTensors;
    std::cout << "inp:" << numInputTensors << " oup:" << numOutputTensors << std::endl;
    cudlaModuleTensorDescriptor *input_tensor_desc = (cudlaModuleTensorDescriptor *)
        malloc(sizeof(cudlaModuleTensorDescriptor) * numInputTensors);

    cudlaModuleTensorDescriptor *output_tensor_desc = (cudlaModuleTensorDescriptor *)
        malloc(sizeof(cudlaModuleTensorDescriptor) * numOutputTensors);

    if ((input_tensor_desc == nullptr) || (output_tensor_desc == nullptr))
    {
      std::runtime_error("cannot request mem for input and output");
    }
    resource.inputTensorDesc = input_tensor_desc;
    resource.outputTensorDesc = output_tensor_desc;
    std::cout << input_tensor_desc << std::endl;

    attribute.inputTensorDesc = input_tensor_desc;
    NVDLA_CHECK_RETURN(cudlaModuleGetAttributes(module_handler, CUDLA_INPUT_TENSOR_DESCRIPTORS, &attribute));
    attribute.outputTensorDesc = output_tensor_desc;
    NVDLA_CHECK_RETURN(cudlaModuleGetAttributes(module_handler, CUDLA_OUTPUT_TENSOR_DESCRIPTORS, &attribute));
    //
    return true;
  }

  // class  DLAConverter
  DLAConverter::DLAConverter(const char* onnx, utils::WorkSpace wSpace) 
  {
    auto trtBuilder = trt::EngineBuilder(wSpace);
    // step1: onnx -> dla
    trtBuilder.build(onnx);
    // step2: use cudla forward
    auto trt_model = trtBuilder.engineName;
    memset(&resource_, 0x00, sizeof(resource_));
    // load trt to resource
    bool load_flag = cudlaLoadTRT(trt_model.c_str(), resource_);
    if (!load_flag) {
      std::cout << "cudla Failed load  from trt engine" << std::endl;
    }
    // cudla forward 
    std::cout << resource_.inputTensorDesc << std::endl;
    this->cudla_forward(trtBuilder.wSpace.nbInps, trtBuilder.wSpace.nbOups);
  };

  DLAConverter::~DLAConverter() {};

  bool DLAConverter::cudla_forward(int numInps, int numOups) {
    // cuda malloc
    predicitonBindings.resize(numInps + numOups);
    for (int i = 0; i < numInps; i++)
    {
      std::cout << i << std::endl;
      CUDA_CHECK_RETURN(cudaMalloc(&predicitonBindings[i], resource_.inputTensorDesc[i].size));
    }
    for (int j = 0; j < numOups; j++)
    {
      std::cout << resource_.outputTensorDesc[j].size << std::endl;
      CUDA_CHECK_RETURN(cudaMalloc(&predicitonBindings[numInps + j], resource_.outputTensorDesc[j].size));
    }
    // cudla mem register
    resource_.inp_regptrs.resize(numInps);
    resource_.oup_regptrs.resize(numOups);

    for (int i = 0; i < numInps; i++)
    {
      NVDLA_CHECK_RETURN(cudlaMemRegister(resource_.dla_handler, (uint64_t *)predicitonBindings[i],
                                          resource_.inputTensorDesc[i].size,
                                          &resource_.inp_regptrs[i], 0))
    }

    for (int i = 0; i < numOups; i++)
    {
      NVDLA_CHECK_RETURN(cudlaMemRegister(resource_.dla_handler, (uint64_t *)predicitonBindings[numInps + i],
                                          resource_.outputTensorDesc[i].size,
                                          &resource_.oup_regptrs[i], 0))
    }


    // check dimension
    if (this->verbose)
    {
      for (int i = 0; i < numInps; i++)
      {
        printTensorDesc(resource_.inputTensorDesc + i);
      }
      for (int i = 0; i < numOups; i++)
      {
        printTensorDesc(resource_.outputTensorDesc + i);
      }
    }
    // send cudla task
    std::cout << "Send cudla task" << std::endl;
    cudlaTask task;
    task.moduleHandle = resource_.module_handler;
    task.outputTensor = resource_.oup_regptrs.data();
    task.inputTensor = resource_.inp_regptrs.data();
    task.numOutputTensors = numOups; // it should equal to attribute.numOutputTensors
    task.numInputTensors = numInps;
    task.waitEvents = nullptr;
    task.signalEvents = nullptr;
    auto timer = utils::GPUTimer();
    NVDLA_CHECK_RETURN(cudlaSubmitTask(resource_.dla_handler, &task, 1, NULL, 0));
    auto cost_time = timer.end();
    std::cout << "Cudla Runtime: " << cost_time << std::endl;
    return true;
  }
}