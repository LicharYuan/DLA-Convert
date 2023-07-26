  
#include "utils.hpp"
#include <stdexcept>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

namespace utils
{

  void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err)
  {
    if (err == cudaSuccess) return;
    std::cout << statement << " returned " << cudaGetErrorString(err) << "("
              << err << ") at " << file << ":" << line << "\n";
    exit(-1);
  };

  void CheckTRTSucc(const char* file, unsigned line, bool succ)
  {
    if (!succ) {
      std::cout  << " returned error AT "   << file << ":" << line << "\n";
      throw std::runtime_error("TRT  FAILED");
    }
  };

  std::string getFilename(std::string filename) 
  {
    return  std::filesystem::path(filename).stem();
  };


  bool fileExist(const std::string &filepath)
  {
    std::ifstream f(filepath.c_str());
    return f.good();
  }





} // namespace utils



  