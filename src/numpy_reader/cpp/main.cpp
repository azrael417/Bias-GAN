#include "numpy_reader.h"

int main(int argc, char* argv[]){

  std::string filename = "../tests/arr_linear.npy";
  NumpyReader npr(false, -1);

  //init file
  npr.Initile(filename);
  npr.ParseFile(filename);
  
  return 0;
}
