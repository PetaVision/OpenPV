/*
 * RecvPost.cu
 *
 *  Created on: Aug 5, 2014
 *      Author: Sheng Lundquist
 */

#ifndef CUDAUPDATESTATEFUNCTION_HPP_
#define CUDAUPDATESTATEFUNCTION_HPP_

#include "../arch/cuda/CudaKernel.hpp"
#include "../arch/cuda/CudaBuffer.hpp"
#include <assert.h>

namespace PVCuda{
#include <builtin_types.h>

   //Parameter structur
   struct HyPerLCAParams{
      int nbatch;
      int numNeurons;
      int nx;
      int ny;
      int nf;
      int lt;
      int rt;
      int dn;
      int up;
      int numChannels;

      float * V;
      int numVertices;
      float * verticesV;
      float * verticesA;
      float * slopes;
      bool selfInteract;
      double * dtAdapt;
      float tau;
      float * GSynHead;
      float * activity;
   };

  struct ISTAParams{
    int nbatch;
    int numNeurons;
    int nx;
    int ny;
    int nf;
    int lt;
    int rt;
    int dn;
    int up;
    int numChannels;

    float * V;
    float Vth;
    float AMax;
    float AMin;
    float AShift;
    float VWidth;
    bool selfInteract;
    double * dtAdapt;
    float tau;
    float * GSynHead;
    float * activity;
  };


class CudaUpdateHyPerLCALayer : public CudaKernel {
public:
   CudaUpdateHyPerLCALayer(CudaDevice* inDevice);
   
   virtual ~CudaUpdateHyPerLCALayer();

   void setArgs(
      const int nbatch,
      const int numNeurons,
      const int nx,
      const int ny,
      const int nf,
      const int lt,
      const int rt,
      const int dn,
      const int up,
      const int numChannels,

      /* float* */ CudaBuffer* V,

      const int numVertices,
      /* float* */ CudaBuffer* verticesV,
      /* float* */ CudaBuffer* verticesA,
      /* float* */ CudaBuffer* slopes,
      const bool selfInteract,
      /* double* */ CudaBuffer* dtAdapt,
      const float tau,

      /* float* */ CudaBuffer* GSynHead,
      /* float* */ CudaBuffer* activity
   );

protected:
   //This is the function that should be overwritten in child classes
   virtual int do_run();

private:
   HyPerLCAParams params;
};


class CudaUpdateISTALayer : public CudaKernel {
public:
  CudaUpdateISTALayer(CudaDevice* inDevice);
  
  virtual ~CudaUpdateISTALayer();

  void setArgs(
	       const int nbatch,
	       const int numNeurons,
	       const int nx,
	       const int ny,
	       const int nf,
	       const int lt,
	       const int rt,
	       const int dn,
	       const int up,
	       const int numChannels,

	       /* float* */ CudaBuffer* V,
	       
	       const float Vth,
	       /* double* */ CudaBuffer* dtAdapt,
	       const float tau,

	       /* float* */ CudaBuffer* GSynHead,
	       /* float* */ CudaBuffer* activity
	       );

protected:
  //This is the function that should be overwritten in child classes                                                  
  virtual int do_run();

private:
  ISTAParams params;
};


} /* namespace PVCuda */

#endif /* CLKERNEL_HPP_ */
