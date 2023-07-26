#include "dla.hpp"

int main(int argc, char *argv[]){
    std::cout << "[dla converter]" << std::endl 
              << "./dla_fp16 $ONNX_MODEL $BS(optional)" 
              << std::endl;
    utils::WorkSpace work_space;
    const char *onnx_path = argv[1];
    work_space.wINT8 = false;
    std::cout << "Load Onnx:" << onnx_path << std::endl;
    if (argc == 3) {
      int batch_size = std::atoi(argv[2]); 
      std::cout << "Batch Size:" << batch_size << std::endl;
      work_space.batchSize = batch_size;
    }
    
    auto converter = dla::DLAConverter(
      onnx_path, work_space
    );
    std::cout << "[Convert Success ]" << std::endl;

}