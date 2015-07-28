/*
 * ANNLayer.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#include "ANNLayer.hpp"
#include "../layers/updateStateFunctions.h"
#include <limits>

namespace PV {

ANNLayer::ANNLayer() {
   initialize_base();
}

ANNLayer::ANNLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
//#ifdef PV_USE_OPENCL
//   if(gpuAccelerateFlag)
//      initializeGPU();
//#endif
}  // end ANNLayer::ANNLayer(const char *, HyPerCol *)

ANNLayer::~ANNLayer() {}

int ANNLayer::initialize_base() {
   return PV_SUCCESS;
}

int ANNLayer::initialize(const char * name, HyPerCol * hc) {
   int status = PtwiseLinearTransferLayer::initialize(name, hc);

//#ifdef PV_USE_OPENCL
//   numEvents=NUM_ANN_EVENTS;
//#endif
   return status;
}

int ANNLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerLayer::ioParamsFillGroup(ioFlag);
   ioParam_VThresh(ioFlag);
   ioParam_AMin(ioFlag);
   ioParam_AMax(ioFlag);
   ioParam_AShift(ioFlag);
   ioParam_VWidth(ioFlag);
   
   // Set verticesV, verticesA, slopeNegInf, slopeNegPos based on VThresh, AMin, AMax, AShift, VWidth
   if (ioFlag == PARAMS_IO_READ && setVertices()!=PV_SUCCESS) { status = PV_FAILURE; }

   ioParam_clearGSynInterval(ioFlag);
   return status;
}

void ANNLayer::ioParam_VThresh(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "VThresh", &VThresh, -max_pvvdata_t);
}

// Parameter VMin was made obsolete in favor of AMin on July 24, 2015
void ANNLayer::ioParam_AMin(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ && parent->parameters()->present(name, "VMin")) {
      if (parent->columnId()==0) {
         fprintf(stderr, "Error: %s \"%s\" parameter \"VMin\" is obsolete.  Use AMin instead.\n",
               parent->parameters()->groupKeywordFromName(name), name);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
      return;
   }
   parent->ioParamValue(ioFlag, name, "AMin", &AMin, VThresh);
}

// Parameter VMax was made obsolete in favor of AShift on July 24, 2015
void ANNLayer::ioParam_AMax(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ && parent->parameters()->present(name, "VMax")) {
      if (parent->columnId()==0) {
         fprintf(stderr, "Error: %s \"%s\" parameter \"VMax\" is obsolete.  Use AMax instead.\n",
               parent->parameters()->groupKeywordFromName(name), name);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
      return;
   }
   parent->ioParamValue(ioFlag, name, "AMax", &AMax, max_pvvdata_t);
}

// Parameter VShift was made obsolete in favor of AShift on July 24, 2015
void ANNLayer::ioParam_AShift(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ && parent->parameters()->present(name, "VShift")) {
      if (parent->columnId()==0) {
         fprintf(stderr, "Error: %s \"%s\" parameter \"VShift\" is obsolete.  Use AShift instead.\n",
               parent->parameters()->groupKeywordFromName(name), name);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
      return;
   }
   parent->ioParamValue(ioFlag, name, "AShift", &AShift, (pvdata_t) 0);
}

void ANNLayer::ioParam_VWidth(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "VWidth", &VWidth, (pvdata_t) 0);
}

//#ifdef PV_USE_OPENCL
///**
// * Initialize OpenCL buffers.  This must be called after PVLayer data have
// * been allocated.
// */
//int ANNLayer::allocateThreadBuffers(const char * kernel_name)
//{
//   int status = HyPerLayer::allocateThreadBuffers(kernel_name);
//
//   //right now there are no ANN layer specific buffers...
//   return status;
//}
//
//int ANNLayer::initializeThreadKernels(const char * kernel_name)
//{
//   char kernelPath[256];
//   char kernelFlags[256];
//
//   int status = CL_SUCCESS;
//   CLDevice * device = parent->getCLDevice();
//
//   const char * pvRelPath = "../PetaVision";
//   sprintf(kernelPath, "%s/%s/src/kernels/%s.cl", parent->getSrcPath(), pvRelPath, kernel_name);
//   sprintf(kernelFlags, "-D PV_USE_OPENCL -cl-fast-relaxed-math -I %s/%s/src/kernels/", parent->getSrcPath(), pvRelPath);
//
//   // create kernels
//   //
//   krUpdate = device->createKernel(kernelPath, kernel_name, kernelFlags);
////kernel name should already be set correctly!
////   if (spikingFlag) {
////      krUpdate = device->createKernel(kernelPath, kernel_name, kernelFlags);
////   }
////   else {
////      krUpdate = device->createKernel(kernelPath, "Retina_nonspiking_update_state", kernelFlags);
////   }
//
//   int argid = 0;
//
//   status |= krUpdate->setKernelArg(argid++, getNumNeurons());
//   status |= krUpdate->setKernelArg(argid++, clayer->loc.nx);
//   status |= krUpdate->setKernelArg(argid++, clayer->loc.ny);
//   status |= krUpdate->setKernelArg(argid++, clayer->loc.nf);
//   status |= krUpdate->setKernelArg(argid++, clayer->loc.nb);
//
//
//   status |= krUpdate->setKernelArg(argid++, clV);
//   status |= krUpdate->setKernelArg(argid++, VThresh);
//   status |= krUpdate->setKernelArg(argid++, AMax);
//   status |= krUpdate->setKernelArg(argid++, AMin);
//   status |= krUpdate->setKernelArg(argid++, AShift);
//   status |= krUpdate->setKernelArg(argid++, getChannelCLBuffer());
////   status |= krUpdate->setKernelArg(argid++, getChannelCLBuffer(CHANNEL_EXC));
////   status |= krUpdate->setKernelArg(argid++, getChannelCLBuffer(CHANNEL_INH));
//   status |= krUpdate->setKernelArg(argid++, clActivity);
//
//   return status;
//}
//int ANNLayer::updateStateOpenCL(double time, double dt)
//{
//   int status = CL_SUCCESS;
//
//   // wait for memory to be copied to device
//   if (numWait > 0) {
//       status |= clWaitForEvents(numWait, evList);
//   }
//   for (int i = 0; i < numWait; i++) {
//      clReleaseEvent(evList[i]);
//   }
//   numWait = 0;
//
//   status |= krUpdate->run(getNumNeurons(), nxl*nyl, 0, NULL, &evUpdate);
//   krUpdate->finish();
//
//   status |= getChannelCLBuffer()->copyFromDevice(1, &evUpdate, &evList[getEVGSyn()]);
//   status |= clActivity->copyFromDevice(1, &evUpdate, &evList[getEVActivity()]);
//   numWait += 2; //3;
//
//
//   return status;
//}
//#endif

int ANNLayer::setVertices() {
   if (VWidth<0) {
      VThresh += VWidth;
      VWidth = -VWidth;
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" warning: interpreting negative VWidth as setting VThresh=%f and VWidth=%f\n",
               parent->parameters()->groupKeywordFromName(name), name, VThresh, VWidth);
      }
   }

   pvdata_t limfromright = VThresh+VWidth-AShift;
   if (AMax < limfromright) limfromright = AMax;

   if (AMin > limfromright) {
      if (parent->columnId()==0) {
         if (VWidth==0) {
            fprintf(stderr, "%s \"%s\" warning: nonmonotonic transfer function, jumping from %f to %f at Vthresh=%f\n",
                  parent->parameters()->groupKeywordFromName(name), name, AMin, limfromright, VThresh);
         }
         else {
            fprintf(stderr, "%s \"%s\" warning: nonmonotonic transfer function, changing from %f to %f as V goes from VThresh=%f to VThresh+VWidth=%f\n",
                  parent->parameters()->groupKeywordFromName(name), name, AMin, limfromright, VThresh, VThresh+VWidth);
         }
      }
   }
   
   // Initialize slopes to NaN so that we can tell whether they've been initialized.
   slopeNegInf = std::numeric_limits<double>::quiet_NaN();
   slopePosInf = std::numeric_limits<double>::quiet_NaN();
   std::vector<pvpotentialdata_t> vectorV;
   std::vector<pvadata_t> vectorA;
   
   slopePosInf = 1.0f;
   if (VThresh <= -0.999*max_pvadata_t) {
      numVertices = 1;
      vectorV.push_back((pvpotentialdata_t) 0);
      vectorA.push_back(-AShift);
      slopeNegInf = 1.0f;
   }
   else {
      assert(VWidth >= (pvpotentialdata_t) 0);
      if (VWidth == (pvpotentialdata_t) 0 && (pvadata_t) VThresh - AShift == AMin) {  // Should there be a tolerance instead of strict ==?
         numVertices = 1;
         vectorV.push_back(VThresh);
         vectorA.push_back(AMin);
      }
      else {
         numVertices = 2;
         vectorV.push_back(VThresh);
         vectorV.push_back(VThresh+VWidth);
         vectorA.push_back(AMin);
         vectorA.push_back(VThresh+VWidth-AShift);
      }
      slopeNegInf = 0.0f;
   }
   if (AMax < 0.999*max_pvadata_t) {
      assert(slopePosInf == 1.0f);
      if (vectorA[numVertices-1] < AMax) {
         pvadata_t interval = AMax - vectorA[numVertices-1];
         vectorV.push_back(vectorV[numVertices-1]+(pvpotentialdata_t) interval);
         vectorA.push_back(AMax);
         numVertices++;
      }
      else {
         // find the last vertex where A < AMax.
         bool found = false;
         int v;
         for (v=numVertices-1; v>=0; v--) {
            if (vectorA[v] < AMax) { found = true; break; }
         }
         if (found) {
            assert(v+1 < numVertices && vectorA[v] < AMax && vectorA[v+1] >= AMax);
            pvadata_t interval = AMax - vectorA[v];
            numVertices = v+1;
            vectorA.resize(numVertices);
            vectorV.resize(numVertices);
            vectorV.push_back(vectorV[v]+(pvpotentialdata_t) interval);
            vectorA.push_back(AMax);
            // In principle, there could be a case where a vertex n has A[n]>AMax but A[n-1] and A[n+1] are both < AMax.
            // But with the current ANNLayer parameters, that won't happen.
         }
         else {
            // All vertices have A>=AMax.
            // If slopeNegInf is positive, transfer function should increase from -infinity to AMax, and then stays constant.
            // If slopeNegInf is negative or zero, 
               numVertices = 1;
               vectorA.resize(numVertices);
               vectorV.resize(numVertices);
            if (slopeNegInf > 0) {
               pvadata_t intervalA = vectorA[0]-AMax;
               pvpotentialdata_t intervalV = (pvpotentialdata_t) (intervalA / slopeNegInf);
               vectorV[0] = vectorV[0] - intervalV;
               vectorA[0] = AMax;
            } 
            else {
               // Everything everywhere is above AMax, so make the transfer function a constant A=AMax.
               vectorA.resize(1);
               vectorV.resize(1);
               vectorV[0] = (pvpotentialdata_t) 0;
               vectorA[0] = AMax;
               numVertices = 1;
               slopeNegInf = 0;
            }
         }
         
      }
      slopePosInf = 0.0f;
   }
   assert(!isnan(slopeNegInf) && !isnan(slopePosInf));
   verticesV = (pvpotentialdata_t *) malloc((size_t) numVertices * sizeof(*verticesV));
   verticesA = (pvpotentialdata_t *) malloc((size_t) numVertices * sizeof(*verticesV));
   if (verticesV==NULL || verticesA==NULL) {
      fprintf(stderr, "%s \"%s\" error: unable to allocate memory for vertices:%s\n",
            parent->parameters()->groupKeywordFromName(name), name, strerror(errno));
   }
   for (int v=0; v<numVertices; v++) {
      verticesV[v] = vectorV[v];
      verticesA[v] = vectorA[v];
      if (parent->columnId()==0) {
         printf("Layer \"%s\": vertex %d, V=%f, A=%f\n", name, v+1, vectorV[v], vectorA[v]);
      }
   }
   // TODO: find a nonstupid way to get the information into the member variable arrays.
   
   return PV_SUCCESS;
}

int ANNLayer::checkVertices() {
   int status = PtwiseLinearTransferLayer::checkVertices(); // checks that the V-coordinates are nondecreasing.
   // TODO: Warn if A is nondecreasing?
   for (int v=1; v < numVertices; v++) { 
      if (verticesA[v] < verticesA[v-1]) {
         if (this->getParent()->columnId()==0) {
            fprintf(stderr, "%s \"%s\" warning: vertices %d and %d: A-coordinates decrease from %f to %f.\n",
                  this->getParent()->parameters()->groupKeywordFromName(this->getName()),
                  this->getName(), v, v+1, verticesA[v-1], verticesA[v]);
         }
      }
   }
   return status;
}

int ANNLayer::resetGSynBuffers(double timef, double dt) {
   int status = PV_SUCCESS;
   if (GSyn == NULL) return PV_SUCCESS;
   bool clearNow = clearGSynInterval <= 0 || timef >= nextGSynClearTime;
   if (clearNow) {
      resetGSynBuffers_HyPerLayer(this->getNumNeurons(), getNumChannels(), GSyn[0]);
   }
   if (clearNow > 0) {
      nextGSynClearTime += clearGSynInterval;   
   }
   return status;
}

int ANNLayer::doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
      pvdata_t * V, int num_channels, pvdata_t * gSynHead)
{
   return PtwiseLinearTransferLayer::doUpdateState(time, dt, loc, A, V, num_channels, gSynHead);
}

int ANNLayer::setActivity() {
   const PVLayerLoc * loc = getLayerLoc();
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   PVHalo const * halo = &loc->halo;
   int num_neurons = nx*ny*nf;
   int status;
   status = setActivity_HyPerLayer(num_neurons, getCLayer()->activity->data, getV(), nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);
   if( status == PV_SUCCESS ) status = applyVThresh_ANNLayer(num_neurons, getV(), AMin, VThresh, AShift, VWidth, getCLayer()->activity->data, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);
   if( status == PV_SUCCESS ) status = applyVMax_ANNLayer(num_neurons, getV(), AMax, getCLayer()->activity->data, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);
   return status;
}

int ANNLayer::checkpointRead(char const * cpDir, double * timeptr) {
   int status = HyPerLayer::checkpointRead(cpDir, timeptr);
   if (status==PV_SUCCESS) {
      status = parent->readScalarFromFile(cpDir, getName(), "nextGSynClearTime", &nextGSynClearTime, parent->simulationTime()-parent->getDeltaTime());
   }
   return status;
}

int ANNLayer::checkpointWrite(char const * cpDir) {
   int status = HyPerLayer::checkpointWrite(cpDir);
   if (status==PV_SUCCESS) {
      status = parent->writeScalarToFile(cpDir, getName(), "nextGSynClearTime", nextGSynClearTime);
   }
   return status;
}


}  // end namespace PV
