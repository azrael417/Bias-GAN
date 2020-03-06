#ifndef __NUMPY_READER_H__
#define __NUMPY_READER_H__

//cuda stuff
#include <cuda.h>
#include <cuda_runtime.h>
#include "cufile.h"

//torch extension
#include <torch/extension.h>

//pread stuff
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

//standard includes
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <map>
#include <string>
#include <assert.h>
#include <algorithm>
#include <regex>

//threading
#include <thread>

//exception handling
#include <exception>
#include <stdexcept>

class NumpyReader {
public:
  
  explicit NumpyReader(const bool& split_axis, const int& device) :
    _num_inter_threads(1),
    _num_intra_threads(1),
    _header_regex(R"###(^\{'descr': \'(.*?)\', 'fortran_order': (.*?), 'shape': \((.*?)\), \})###"),
    _device(torch::kCPU),
      _split_axis(split_axis),
      _fortran_order(false),
      _little_endian(true),
      _numsample(0),
      _numelem(0),
      _typesize(0),
      _batchsize(1),
      _data(nullptr),
      _ddata(nullptr),
      _teptr(nullptr)
		 {
       //set up typemap
       _typemap = {
           {"i8", torch::kInt64},
           {"i4", torch::kInt32},
           {"f8", torch::kFloat64},
           {"f4", torch::kFloat32}
       };

       if( device >= 0 ){
         _device = torch::Device(torch::kCUDA, device);
       }
       
       if( _device.is_cuda() ){
         //set device
         cudaSetDevice(_device.index());
         //open cufile driver
         _cf_status = cuFileDriverOpen();
         if (_cf_status.err!= CU_FILE_SUCCESS) {
           throw std::runtime_error("NumpyReader: cuFile driver failed to open");
         }
       }
		 }
     
  ~NumpyReader(){
    if(_data != nullptr) delete [] _data;
    if(_ddata != nullptr){
      //de-register dev buffer
      _cf_status = cuFileBufDeregister(_ddata);
      //free buffer
      cudaFree(_ddata);
    }
    if( _device.is_cuda() ){
      cuFileDriverClose();
    }
  }
  
  //set batch size:
  void SetBatchsize(const unsigned int& batch_size);

  //prep and finish IO
  void InitFile(const std::string& filename);
  void FinalizeFile();
  
  //actual iterator
  torch::Tensor getSample( int64_t idx );
  torch::Tensor getBatch( const std::vector<int64_t>& idxs );
  
  //called by the class
  void ParseFile(const std::string& filename);
  
  //print info convenience functions
  void PrintHeaderInfo();

  //some accessor functions
  int64_t getNumSamples() const{
    return _numsample;
  }
  
  //return the shape vector
  std::vector<int64_t> getShape() const{
    return _shape;
  }
  
  //return the stripe vector
  std::vector<int64_t> getStrides() const{
    return _stride;
  }

  //variables
  unsigned int _num_inter_threads;
  unsigned int _num_intra_threads;
  
private:
  //regex search
  const std::regex _header_regex;
	
  //external variables stored
  std::vector<int64_t> _shape;
  std::vector<int64_t> _stride;
  torch::Dtype _type;
  torch::Device _device;
  bool _split_axis;
  
  //other variables
  int _fd;
  bool _fortran_order;
  bool _little_endian;
  int64_t _numsample;
  int64_t _numelem;
  int64_t _typesize;
  int64_t _batchsize;
  off_t _offset;
  unsigned char *_data, *_ddata;
  torch::Tensor _sample;
  
  //cufile stuff
  CUfileError_t _cf_status;
  CUfileDescr_t _cf_descr;
  CUfileHandle_t _cf_handle;
  
  //typemap for checking
  std::map<std::string, torch::Dtype> _typemap;
  
  //exception handling
  std::exception_ptr _teptr;
	
  //compute the tensor stride based on fortran order or not
  void computeStrides();
  
  //allocate memory: only called by parse header
  void allocate();
  
  //read one item from src into dst
  void readSample(int64_t dst_idx, int64_t src_idx);

  //more lower level read routine
  void readChunk(int64_t dest_off, int64_t src_off, int64_t size);
};


#endif  //__NUMPY_READER_H__
