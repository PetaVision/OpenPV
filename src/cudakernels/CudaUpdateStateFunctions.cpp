#include "CudaUpdateStateFunctions.hpp"


namespace PVCuda{

CudaUpdateHyPerLCALayer::CudaUpdateHyPerLCALayer(CudaDevice* inDevice):CudaKernel(inDevice){
   kernelName = "HyPerLCALayer";
}
  
CudaUpdateHyPerLCALayer::~CudaUpdateHyPerLCALayer(){
}

CudaUpdateMomentumLCALayer::CudaUpdateMomentumLCALayer(CudaDevice* inDevice):CudaKernel(inDevice){
   kernelName = "MomentumLCALayer";
}
  
CudaUpdateMomentumLCALayer::~CudaUpdateMomentumLCALayer(){
}

CudaUpdateISTALayer::CudaUpdateISTALayer(CudaDevice* inDevice):CudaKernel(inDevice){
   kernelName = "ISTALayer";
}

CudaUpdateISTALayer::~CudaUpdateISTALayer(){
}

void CudaUpdateHyPerLCALayer::setArgs(
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
				      ){
   params.nbatch = nbatch;
   params.numNeurons = numNeurons;
   params.nx = nx;
   params.ny = ny;
   params.nf = nf;
   params.lt = lt;
   params.rt = rt;
   params.dn = dn;
   params.up = up;
   params.numChannels = numChannels;

   params.V = (float*) V->getPointer();

   params.numVertices = numVertices;
   params.verticesV = (float*) verticesV->getPointer();
   params.verticesA = (float*) verticesA->getPointer();
   params.slopes = (float*) slopes->getPointer();
   params.selfInteract = selfInteract;
   params.dtAdapt = (double*) dtAdapt->getPointer();
   params.tau = tau;

   params.GSynHead = (float*) GSynHead->getPointer();
   params.activity = (float*) activity->getPointer();

   setArgsFlag();
}


void CudaUpdateMomentumLCALayer::setArgs(
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
                  /* float* */ CudaBuffer* prevDrive,
				      
				      const int numVertices,
				      /* float* */ CudaBuffer* verticesV,
				      /* float* */ CudaBuffer* verticesA,
				      /* float* */ CudaBuffer* slopes,
				      const bool selfInteract,
				      /* double* */ CudaBuffer* dtAdapt,
				      const float tau,
                  const float LCAMomentumRate,
				      
				      /* float* */ CudaBuffer* GSynHead,
				      /* float* */ CudaBuffer* activity
				      ){
   params.nbatch = nbatch;
   params.numNeurons = numNeurons;
   params.nx = nx;
   params.ny = ny;
   params.nf = nf;
   params.lt = lt;
   params.rt = rt;
   params.dn = dn;
   params.up = up;
   params.numChannels = numChannels;

   params.V = (float*) V->getPointer();
   params.prevDrive = (float*) prevDrive->getPointer();

   params.numVertices = numVertices;
   params.verticesV = (float*) verticesV->getPointer();
   params.verticesA = (float*) verticesA->getPointer();
   params.slopes = (float*) slopes->getPointer();
   params.selfInteract = selfInteract;
   params.dtAdapt = (double*) dtAdapt->getPointer();
   params.tau = tau;
   params.LCAMomentumRate = LCAMomentumRate;
   
   params.GSynHead = (float*) GSynHead->getPointer();
   params.activity = (float*) activity->getPointer();
   
   setArgsFlag();
}


void CudaUpdateISTALayer::setArgs(
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
				  ){
  params.nbatch = nbatch;
  params.numNeurons = numNeurons;
  params.nx = nx;
  params.ny = ny;
  params.nf = nf;
  params.lt = lt;
  params.rt = rt;
  params.dn = dn;
  params.up = up;
  params.numChannels = numChannels;
  
  params.V = (float*) V->getPointer();
  
  params.Vth = Vth;
  params.dtAdapt = (double*) dtAdapt->getPointer();
  params.tau = tau;
  
  params.GSynHead = (float*) GSynHead->getPointer();
  params.activity = (float*) activity->getPointer();
    
  setArgsFlag();
}

}  // end namespace PVCuda
