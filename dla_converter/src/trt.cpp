#include "trt.hpp"
#include <cassert>
#include "NvInfer.h"
#include <sys/stat.h>
#include <filesystem>
#include <fstream>

namespace trt
{
  bool EngineBuilder::build(const char* onnx) {
    auto model_name = getFilename(onnx);
    engineName = model_name + ".dla";
    if (utils::fileExist(engineName)) {
      std::cout << "[WARN] file exits, regenerating ... " << std::endl;
    }
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    builder->setMaxBatchSize(wSpace.maxBatchSize);
    if ((wSpace.maxBatchSize) > builder->getMaxDLABatchSize()){
      std::string msg = "max batchSize is larger than DLA limited" + std::to_string(wSpace.maxBatchSize) + " V.S " + std::to_string(builder->getMaxDLABatchSize());
      throw std::runtime_error(msg);
    }
    auto profileStream = makeCudaStream();
    if (!profileStream) {
      throw std::runtime_error("fail to create stream for profile");
    } 

    auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
      std::cout << " createNetworkV2 failed " << std::endl;
      return false;
    }
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (!parser)
    {
      std::cout << " parse builder failed " << std::endl;
      return false;
    }

    TRT_CHECK_RETURN(parser->parseFromFile(onnx, static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)));
    std::cout << "parsed Onnx Model: " <<  onnx << std::endl;

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
      std::cout << "config build failed" << std::endl;
      return false;
    }
    // set config
    // use dla 0 to generate trt engine
    config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
    config->setDLACore(0); 
    // cudla require no reformats and standalone
    config->setEngineCapability(nvinfer1::EngineCapability::kDLA_STANDALONE);
    config->setFlag(nvinfer1::BuilderFlag::kDIRECT_IO);
    config->setMaxWorkspaceSize(wSpace.maxWorkspaceSize);
    config->setProfileStream(*profileStream);
    if (wSpace.wINT8) {
      config->setFlag(nvinfer1::BuilderFlag::kINT8);
    } else {
      config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    auto *defaultProfiler = builder->createOptimizationProfile();
    // update workspace
    wSpace.nbInps = network->getNbInputs();
    wSpace.nbOups = network->getNbOutputs();
    std::cout << "Update Workspace NbIO from onnx model: inp: " 
              << wSpace.nbInps <<  "oup: " << wSpace.nbOups << std::endl; 
    // set input
    for (int i = 0; i < network->getNbInputs(); i++)
    {
      // set each input profiler
      const auto input = network->getInput(i);
      // input->setDynamicRange(-128., 127.);
      const auto inputDims = input->getDimensions();
      const int32_t inputC = inputDims.d[1];
      const int32_t inputH = inputDims.d[2];
      const int32_t inputW = inputDims.d[3];
      nvinfer1::Dims32 dim32{4, {wSpace.batchSize, inputC, inputH, inputW}};
      network->getInput(i)->setDimensions(dim32);

      // DLA only support particular IO format
      // More infomration can be found in:
      //   https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-standalone-mode
      if (wSpace.wINT8) {
        input->setType(nvinfer1::DataType::kINT8);
        input->setAllowedFormats(
          nvinfer1::TensorFormats(
            // 1U << static_cast<int>(TensorFormat::kDLA_LINEAR) |
            1U << static_cast<int>(nvinfer1::TensorFormat::kCHW32)));
        
      } else  {
        input->setType(nvinfer1::DataType::kHALF);
        input->setAllowedFormats(
          nvinfer1::TensorFormats(1U << static_cast<int>(nvinfer1::TensorFormat::kCHW16)));
      }
    }

    // set output
    for (int i = 0; i < network->getNbOutputs(); i++)
    {
      const auto output = network->getOutput(i);
      if (wSpace.wINT8) {
        output->setType(nvinfer1::DataType::kINT8);
        output->setAllowedFormats(nvinfer1::TensorFormats(1U << static_cast<int>(nvinfer1::TensorFormat::kCHW32)));
      } else  {
        output->setType(nvinfer1::DataType::kHALF);
        output->setAllowedFormats(nvinfer1::TensorFormats(1U << static_cast<int>(nvinfer1::TensorFormat::kCHW16)));
      }
    }

    // load calib
    if (wSpace.wINT8) {
      auto calibMap = getCalib(wSpace.calib_file);
      printCalib(calibMap);
      // set dynamic range  for
      for (int i = 0; i < network->getNbInputs(); ++i)
      {
        auto tName = network->getInput(i)->getName();
        if (calibMap.find(tName) != calibMap.end())
        {
          if (!network->getInput(i)->setDynamicRange(-calibMap.at(tName), calibMap.at(tName)))
          {
            throw std::runtime_error("Unable set Input Dynamic Range");
          }
        } else {
          std::cout << "input name:" << tName << "is not found in calibration" << std::endl;
          throw std::runtime_error("calibration is not correct");
        }
      }
      // set each layer output
      for (int i = 0; i < network->getNbLayers(); ++i) 
      {
        auto lyr = network->getLayer(i);
        for (int j = 0, e = lyr->getNbOutputs(); j < e; ++j) 
        {
          auto tName = lyr->getOutput(j)->getName();
          if (calibMap.find(tName) != calibMap.end()) 
          {
            if (!lyr->getOutput(j)->setDynamicRange(-calibMap.at(tName), calibMap.at(tName))) 
            {
              throw std::runtime_error("cannot set dynamic range for layer oup");
            }
          } else {
            std::cout << "layer out name:"  << tName << "is not found in calibration" << std::endl;
            throw std::runtime_error("calibration is not correct");
          }
        }
      }
    }
    // build engine
    auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan)
    {
      throw std::runtime_error("faile to seralize engine");
    }
    std::ofstream outfile(engineName, std::ios::out | std::ios::binary);
    outfile.write(reinterpret_cast<const char *>(plan->data()), plan->size());
    return true;
 
  }

  std::unordered_map<std::string, float> getCalib(std::string files) {
    assert(utils::fileExist(files));
    std::unordered_map<std::string, float> tensor_dynamic_range;
    std::ifstream calib_file(files);
    std::string line;
    char delim = ':';
    if (calib_file.is_open()) {
      while (getline(calib_file, line)) {
        if (line.find(delim) != std::string::npos) {
          std::istringstream iline(line);
          std::string token;
          getline(iline, token, delim);
          std::string tensor_name = token;
          getline(iline, token, delim);
          uint32_t tmp_range;
          // read calibration scale
          // see https://github.com/nvdla/sw/blob/master/umd/utils/calibdata/calib_txt_to_json.py#L42
          sscanf(token.c_str(), "%x", &tmp_range);
          float dynamic_range = *((float *)&tmp_range);
          // NOTE: should make sure the quant range is [-127, 127]
          tensor_dynamic_range[tensor_name] = 127. * dynamic_range;
        }
      }
    }
    return tensor_dynamic_range;
  }

  void printCalib(std::unordered_map<std::string, float> map) {
    for (const auto& elem: map ) {
      std::cout << elem.first << ": " << elem.second << std::endl;
    }
  }

  

} // namespace trt
