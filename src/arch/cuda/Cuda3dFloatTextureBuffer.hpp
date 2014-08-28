///*
// * Cuda3dFloatTextureBuffer.hpp
// *
// *  Created on: Aug 11, 2014
// *      Author: Sheng Lundquist
// */
//
//#ifndef CUDA3DFLOATTEXTUREBUFFER_HPP_
//#define CUDA3DFLOATTEXTUREBUFFER_HPP_
//
//#include "CudaBuffer.hpp"
//#include <cuda.h>
//
//////////////////////////////////////////////////////////////////////////////////
//
//namespace PVCuda {
//
//class Cuda3dFloatTextureBuffer:public CudaBuffer {
//
//public:
//
//   Cuda3dFloatTextureBuffer(int num_size_x, int num_size_y, int num_size_z);
//   virtual ~Cuda3dFloatTextureBuffer();
//   
//   virtual int copyToDevice(void * h_ptr);
//   virtual int copyFromDevice(void* h_ptr);
//
//   virtual cudaArray* getArrayPointer(){return cudaarray;}
//   virtual cudaChannelFormatDesc getChannel(){return channel;}
//   
//
//protected:
//
//private:
//   cudaArray* cudaarray;
//   cudaExtent volumesize;
//   cudaChannelFormatDesc channel;
//   cudaMemcpy3DParms copyparams;
//
//   int num_size_x;
//   int num_size_y;
//   int num_size_z;
//};
//
//} // namespace PV
//
//#endif
