/*
 * PtwiseLinearTransferLayer.cpp
 *
 *  Created on: July 24, 2015
 *      Author: pschultz
 */

#include "PtwiseLinearTransferLayer.hpp"
#include "updateStateFunctions.h"
#include <limits>

#ifdef __cplusplus
extern "C" {
#endif

void PtwiseLinearTransferLayer_update_state(
    const int nbatch,
    const int numNeurons,
    const int nx,
    const int ny,
    const int nf,
    const int lt,
    const int rt,
    const int dn,
    const int up,

    float * V,
    int numVertices,
    float * verticesV,
    float * verticesA,
    float * slopes,
    int num_channels,
    float * GSynHead,
    float * activity);

#ifdef __cplusplus
}
#endif

namespace PV {

PtwiseLinearTransferLayer::PtwiseLinearTransferLayer() {
   initialize_base();
}

PtwiseLinearTransferLayer::PtwiseLinearTransferLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
//#ifdef PV_USE_OPENCL
//   if(gpuAccelerateFlag)
//      initializeGPU();
//#endif
}  // end PtwiseLinearTransferLayer::PtwiseLinearTransferLayer(const char *, HyPerCol *)

PtwiseLinearTransferLayer::~PtwiseLinearTransferLayer() {
   free(verticesV); verticesV = NULL;
   free(verticesA); verticesA = NULL;
   free(slopes); slopes = NULL;
}

int PtwiseLinearTransferLayer::initialize_base() {
   numVertices = 0;
   verticesV = NULL;
   verticesA = NULL;
   slopes = NULL;
   slopeNegInf = 1.0f;
   slopePosInf = 1.0f;
   return PV_SUCCESS;
}

int PtwiseLinearTransferLayer::initialize(const char * name, HyPerCol * hc) {
   int status = HyPerLayer::initialize(name, hc);
   assert(status == PV_SUCCESS);

   status |= checkVertices();
   status |= setSlopes();
//#ifdef PV_USE_OPENCL
//   numEvents=NUM_ANN_EVENTS;
//#endif
   return status;
}

int PtwiseLinearTransferLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerLayer::ioParamsFillGroup(ioFlag);

   if (ioFlag==PARAMS_IO_READ) { assert(numVertices==0); }
   ioParam_verticesV(ioFlag);
   ioParam_verticesA(ioFlag);
   assert(numVertices!=0);

   ioParam_slopeNegInf(ioFlag);
   ioParam_slopePosInf(ioFlag);

   ioParam_clearGSynInterval(ioFlag);

   return status;
}

void PtwiseLinearTransferLayer::ioParam_verticesV(enum ParamsIOFlag ioFlag) {
   int numVerticesTmp = numVertices;
   this->getParent()->ioParamArray(ioFlag, this->getName(), "verticesV", &verticesV, &numVerticesTmp);
   if (ioFlag==PARAMS_IO_READ) {
      if (numVerticesTmp==0) {
         if (this->getParent()->columnId()==0) {
            fprintf(stderr,
                  "%s \"%s\" error: verticesV cannot be empty\n",
                  this->getKeyword(), this->getName());
         }
         MPI_Barrier(this->getParent()->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      if (numVertices !=0 && numVerticesTmp != numVertices) {
         if (this->getParent()->columnId()==0) {
            fprintf(stderr,
                  "%s \"%s\" error: verticesV (%d elements) and verticesA (%d elements) must have the same lengths.\n",
                  this->getKeyword(), this->getName(), numVerticesTmp, numVertices);
         }
         MPI_Barrier(this->getParent()->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      assert(numVertices==0 || numVertices==numVerticesTmp);
      numVertices = numVerticesTmp;
   }
}

void PtwiseLinearTransferLayer::ioParam_verticesA(enum ParamsIOFlag ioFlag) {
   int numVerticesA;
   this->getParent()->ioParamArray(ioFlag, this->getName(), "verticesA", &verticesA, &numVerticesA);
   if (ioFlag==PARAMS_IO_READ) {
      if (numVerticesA==0) {
         if (this->getParent()->columnId()==0) {
            fprintf(stderr,
                  "%s \"%s\" error: verticesA cannot be empty\n",
                  this->getKeyword(), this->getName());
         }
         MPI_Barrier(this->getParent()->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      if (numVertices !=0 && numVerticesA != numVertices) {
         if (this->getParent()->columnId()==0) {
            fprintf(stderr,
                  "%s \"%s\" error: verticesV (%d elements) and verticesA (%d elements) must have the same lengths.\n",
                  this->getKeyword(), this->getName(), numVertices, numVerticesA);
         }
         MPI_Barrier(this->getParent()->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      assert(numVertices==0 || numVertices==numVerticesA);
      numVertices = numVerticesA;
   }
}

void PtwiseLinearTransferLayer::ioParam_slopeNegInf(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "slopeNegInf", &slopeNegInf, slopeNegInf/*default*/, true/*warnIfAbsent*/);
}

void PtwiseLinearTransferLayer::ioParam_slopePosInf(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "slopePosInf", &slopePosInf, slopePosInf/*default*/, true/*warnIfAbsent*/);
}

void PtwiseLinearTransferLayer::ioParam_clearGSynInterval(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "clearGSynInterval", &clearGSynInterval, 0.0);
   if (ioFlag==PARAMS_IO_READ) {
      nextGSynClearTime = parent->getStartTime();
   }
}

//#ifdef PV_USE_OPENCL
///**
// * Initialize OpenCL buffers.  This must be called after PVLayer data have
// * been allocated.
// */
//int PtwiseLinearTransferLayer::allocateThreadBuffers(const char * kernel_name)
//{
//   int status = HyPerLayer::allocateThreadBuffers(kernel_name);
//
//   //right now there are no PtwiseLinearTransferLayer-specific buffers...
//   return status;
//}
//
//int PtwiseLinearTransferLayer::initializeThreadKernels(const char * kernel_name)
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
//int PtwiseLinearTransferLayer::updateStateOpenCL(double time, double dt)
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

int PtwiseLinearTransferLayer::checkVertices() {
   assert(numVertices>0);
   int status = PV_SUCCESS;
   for (int v=1; v<numVertices; v++) {
      if (verticesV[v] < verticesV[v-1]) {
         status = PV_FAILURE;
         if (this->getParent()->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: vertices %d and %d: V-coordinates decrease from %f to %f.\n",
                  this->getKeyword(), this->getName(), v, v+1, verticesV[v-1], verticesV[v]);
         }
      }
   }
   return status;
}

int PtwiseLinearTransferLayer::setSlopes() {
   assert(numVertices>0);
   assert(verticesA!=NULL);
   assert(verticesV!=NULL);
   slopes = (float *) malloc((size_t)(numVertices+1)*sizeof(*slopes));
   if (slopes == NULL) {
      fprintf(stderr, "%s \"%s\" error: unable to allocate memory for transfer function slopes: %s\n",
            this->getKeyword(), name, strerror(errno));
      exit(EXIT_FAILURE);
      
   }
   slopes[0] = slopeNegInf;
   slopes[numVertices] = slopePosInf;
   for (int k=1; k<numVertices; k++) {
      float V1 = verticesV[k-1];
      float V2 = verticesV[k];
      if (V1!=V2) {
         slopes[k] = (verticesA[k]-verticesA[k-1])/(V2-V1);
      }
      else {
         slopes[k] = verticesA[k]>verticesA[k-1] ? std::numeric_limits<float>::infinity() :
                     verticesA[k]<verticesA[k-1] ? -std::numeric_limits<float>::infinity() :
                     std::numeric_limits<float>::quiet_NaN();
      }
   }
   return PV_SUCCESS;
}

int PtwiseLinearTransferLayer::resetGSynBuffers(double timef, double dt) {
   int status = PV_SUCCESS;
   if (GSyn == NULL) return PV_SUCCESS;
   bool clearNow = clearGSynInterval <= 0 || timef >= nextGSynClearTime;
   if (clearNow) {
      resetGSynBuffers_HyPerLayer(parent->getNBatch(), this->getNumNeurons(), getNumChannels(), GSyn[0]);
   }
   if (clearNow > 0) {
      nextGSynClearTime += clearGSynInterval;   
   }
   return status;
}

//! PtwiseLinearTransferLayer update state function, to add support for GPU kernel.
//
/*!
 * REMARKS:
 *      - The kernel calls PtwiseLinearTransferLayer_update_state in
 *        updateStateFunctions.h, which calls
 *        applyGSyn_HyPerLayer (sets V = GSynExc - GSynInh)
 *        setActivity_PtwiseLinearTransferLayer (computes A from V)
 */
int PtwiseLinearTransferLayer::doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
      pvdata_t * V, int num_channels, pvdata_t * gSynHead)
{
//#ifdef PV_USE_OPENCL
//   if(gpuAccelerateFlag) {
//      updateStateOpenCL(time, dt);
//   }
//   else {
//#endif
      int nx = loc->nx;
      int ny = loc->ny;
      int nf = loc->nf;
      int num_neurons = nx*ny*nf;
      int nbatch = loc->nbatch;
      PtwiseLinearTransferLayer_update_state(nbatch, num_neurons, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up, V, numVertices, verticesV, verticesA, slopes, num_channels, gSynHead, A);

//#ifdef PV_USE_OPENCL
//   }
//#endif

   return PV_SUCCESS;
}

int PtwiseLinearTransferLayer::setActivity() {
   const PVLayerLoc * loc = getLayerLoc();
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int nbatch = loc->nbatch;
   PVHalo const * halo = &loc->halo;
   int num_neurons = nx*ny*nf;
   return setActivity_PtwiseLinearTransferLayer(nbatch, num_neurons, getCLayer()->activity->data, getV(), nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up, numVertices, verticesV, verticesA, slopes);
}

int PtwiseLinearTransferLayer::checkpointRead(char const * cpDir, double * timeptr) {
   int status = HyPerLayer::checkpointRead(cpDir, timeptr);
   if (status==PV_SUCCESS) {
      status = parent->readScalarFromFile(cpDir, getName(), "nextGSynClearTime", &nextGSynClearTime, parent->simulationTime()-parent->getDeltaTime());
   }
   return status;
}

int PtwiseLinearTransferLayer::checkpointWrite(char const * cpDir) {
   int status = HyPerLayer::checkpointWrite(cpDir);
   if (status==PV_SUCCESS) {
      status = parent->writeScalarToFile(cpDir, getName(), "nextGSynClearTime", nextGSynClearTime);
   }
   return status;
}

BaseObject * createPtwiseLinearTransferLayer(char const * name, HyPerCol * hc) {
   return hc ? new PtwiseLinearTransferLayer(name, hc) : NULL;
}

}  // end namespace PV

///////////////////////////////////////////////////////
//
// implementation of PtwiseLinearTransferLayer kernels
//

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PV_USE_OPENCL
#  include "../kernels/PtwiseLinearTransferLayer_update_state.cl"
#else
#  undef PV_USE_OPENCL
#  include "../kernels/PtwiseLinearTransferLayer_update_state.cl"
#  define PV_USE_OPENCL
#endif

#ifdef __cplusplus
}
#endif

