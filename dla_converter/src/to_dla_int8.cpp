#include "dla.hpp"

int main(int argc, char *argv[]){
    std::cout << "[dla converter]" << std::endl 
              << "./dla_int8 $ONNX_MODEL $CALIB_FILE $BS(optional)" 
              << std::endl;
    utils::WorkSpace work_space;
    const char *onnx_path = argv[1];
    const char *calib_path = argv[2];
    work_space.calib_file = calib_path;
    std::cout << "Load Onnx:" << onnx_path << std::endl;
    std::cout << "Load calib:" << calib_path << std::endl;
    if (argc == 4) {
      int batch_size = std::atoi(argv[3]); 
      std::cout << "Batch Size:" << batch_size << std::endl;
      work_space.batchSize = batch_size;
    }
    
    auto converter = dla::DLAConverter(
      onnx_path, work_space
    );
    std::cout << "[Convert Success ]" << std::endl;

}