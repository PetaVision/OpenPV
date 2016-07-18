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
}

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
   // PtwiseLinearTransferLayer was deprecated June 28, 2016.
   pvWarn() << "PtwiseLinearTransferLayer is deprecated.\n";
   pvWarn() << "ANNLayer now has the option of specifying either\n";
   pvWarn() << "verticesV/verticesA/slopeNegInf/slopePosInf\n";
   pvWarn() << "or VThresh/AMin/AMax/AShift/VWidth.\n";
   int status = HyPerLayer::initialize(name, hc);
   assert(status == PV_SUCCESS);

   status |= checkVertices();
   status |= setSlopes();
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
            pvErrorNoExit().printf("%s: verticesV cannot be empty\n",
                  getDescription_c());
         }
         MPI_Barrier(this->getParent()->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      if (numVertices !=0 && numVerticesTmp != numVertices) {
         if (this->getParent()->columnId()==0) {
            pvErrorNoExit().printf("%s: verticesV (%d elements) and verticesA (%d elements) must have the same lengths.\n",
                  getDescription_c(), numVerticesTmp, numVertices);
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
            pvErrorNoExit().printf("%s: verticesA cannot be empty\n",
                  getDescription_c());
         }
         MPI_Barrier(this->getParent()->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      if (numVertices !=0 && numVerticesA != numVertices) {
         if (this->getParent()->columnId()==0) {
            pvErrorNoExit().printf("%s: verticesV (%d elements) and verticesA (%d elements) must have the same lengths.\n",
                  getDescription_c(), numVertices, numVerticesA);
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
int PtwiseLinearTransferLayer::checkVertices() {
   assert(numVertices>0);
   int status = PV_SUCCESS;
   for (int v=1; v<numVertices; v++) {
      if (verticesV[v] < verticesV[v-1]) {
         status = PV_FAILURE;
         if (this->getParent()->columnId()==0) {
            pvErrorNoExit().printf("%s: vertices %d and %d: V-coordinates decrease from %f to %f.\n",
                  getDescription_c(), v, v+1, verticesV[v-1], verticesV[v]);
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
      pvErrorNoExit().printf("%s: unable to allocate memory for transfer function slopes: %s\n",
            getDescription_c(), strerror(errno));
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
int PtwiseLinearTransferLayer::updateState(double time, double dt)
{
   const PVLayerLoc * loc = getLayerLoc();
   pvdata_t * A = clayer->activity->data;
   pvdata_t * V = getV();
   int num_channels = getNumChannels();
   pvdata_t * gSynHead = GSyn == NULL ? NULL : GSyn[0];
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   int nbatch = loc->nbatch;
   PtwiseLinearTransferLayer_update_state(nbatch, num_neurons, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up, V, numVertices, verticesV, verticesA, slopes, num_channels, gSynHead, A);

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
    float * activity)
{
   updateV_PtwiseLinearTransferLayer(nbatch, numNeurons, V, num_channels, GSynHead, activity, numVertices, verticesV, verticesA, slopes, nx, ny, nf, lt, rt, dn, up);
}
