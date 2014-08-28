///*
// * CudaBuffer.cpp
// *
// *  Created on: Aug 11, 2014
// *      Author: Sheng Lundquist
// */
//
//#include "Cuda3dFloatTextureBuffer.hpp"
//#include "cuda_util.h"
//
//
//namespace PVCuda {
//
//Cuda3dFloatTextureBuffer::Cuda3dFloatTextureBuffer(int num_size_x, int num_size_y, int num_size_z)
//{
//   this->num_size_x = num_size_x;
//   this->num_size_y = num_size_y;
//   this->num_size_z = num_size_z;
//   this->size = num_size_x * num_size_y * num_size_z * sizeof(float);
//
//   volumesize = make_cudaExtent(num_size_x, num_size_y, num_size_z);
//   channel = cudaCreateChannelDesc<float>();
//   handleError(cudaMalloc3DArray(&cudaarray, &channel, volumesize));
//   copyparams.extent=volumesize;
//   copyparams.dstArray=cudaarray;
//   copyparams = {0};
//}
//
//Cuda3dFloatTextureBuffer::~Cuda3dFloatTextureBuffer(){
//   cudaFreeArray(cudaarray);
//}
//
//int Cuda3dFloatTextureBuffer::copyToDevice(void* h_ptr){
//   copyparams.kind=cudaMemcpyHostToDevice;
//   copyparams.srcPtr = make_cudaPitchedPtr(h_ptr, sizeof(float)*num_size_x, num_size_y, num_size_z);
//   cudaMemcpy3D(&copyparams);
//   return 0;
//}
//
////Is this really needed? Texture memory is read only, so it's guarenteed not to change
//int Cuda3dFloatTextureBuffer::copyFromDevice(void* h_ptr){
//   //copyparams.extent=volumesize;
//   //copyparams.dstArray=cudaarray;
//   //copyparams.kind=cudaMemcpyDeviceToHost;
//   //copyparams.srcPtr = make_cudaPitchPtr(h_ptr, sizeof(float)*num_size_x*num_size_y*num_size_z);
//   //cudaMemcpy3D(&copyparams);
//   printf("Cannot copy from device of read only texture buffer\n");
//   exit(-1);
//}
//
//}
