/*
 * ANNLayer.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#include "ANNLayer.hpp"
#include "layers/updateStateFunctions.h"
#include <limits>

void ANNLayer_vertices_update_state(
      const int nbatch,
      const int numNeurons,
      const int nx,
      const int ny,
      const int nf,
      const int lt,
      const int rt,
      const int dn,
      const int up,

      float *V,
      int numVertices,
      float *verticesV,
      float *verticesA,
      float *slopes,
      int num_channels,
      float *GSynHead,
      float *activity);

void ANNLayer_threshminmax_update_state(
      const int nbatch,
      const int numNeurons,
      const int nx,
      const int ny,
      const int nf,
      const int lt,
      const int rt,
      const int dn,
      const int up,

      float *V,
      float VThresh,
      float AMin,
      float AMax,
      float AShift,
      float VWidth,
      int num_channels,
      float *GSynHead,
      float *activity);

namespace PV {

ANNLayer::ANNLayer() { initialize_base(); }

ANNLayer::ANNLayer(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

ANNLayer::~ANNLayer() {
   free(verticesV);
   free(verticesA);
   free(slopes);
}

int ANNLayer::initialize_base() {
   // Data members were initialized in the class member-declarations
   return PV_SUCCESS;
}

int ANNLayer::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   if (!this->layerListsVerticesInParams()) {
      if (status == PV_SUCCESS) {
         status = setVertices();
      }
   }
   if (status == PV_SUCCESS) {
      status = checkVertices();
   }
   if (status == PV_SUCCESS) {
      setSlopes();
   }
   return status;
}

int ANNLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerLayer::ioParamsFillGroup(ioFlag);

   if (parent->parameters()->arrayPresent(name, "verticesV")) {
      verticesListInParams = true;
      ioParam_verticesV(ioFlag);
      ioParam_verticesA(ioFlag);
      ioParam_slopeNegInf(ioFlag);
      ioParam_slopePosInf(ioFlag);
   }
   else {
      verticesListInParams = false;
      ioParam_VThresh(ioFlag);
      ioParam_AMin(ioFlag);
      ioParam_AMax(ioFlag);
      ioParam_AShift(ioFlag);
      ioParam_VWidth(ioFlag);
   }

   return status;
}

void ANNLayer::ioParam_verticesV(enum ParamsIOFlag ioFlag) {
   pvAssert(verticesListInParams);
   int numVerticesTmp = numVertices;
   this->parent->parameters()->ioParamArray(
         ioFlag, this->getName(), "verticesV", &verticesV, &numVerticesTmp);
   if (ioFlag == PARAMS_IO_READ) {
      if (numVerticesTmp == 0) {
         if (this->parent->columnId() == 0) {
            ErrorLog().printf("%s: verticesV cannot be empty\n", getDescription_c());
         }
         MPI_Barrier(this->parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      if (numVertices != 0 && numVerticesTmp != numVertices) {
         if (this->parent->columnId() == 0) {
            ErrorLog().printf(
                  "%s: verticesV (%d elements) and verticesA (%d elements) must have the same "
                  "lengths.\n",
                  getDescription_c(),
                  numVerticesTmp,
                  numVertices);
         }
         MPI_Barrier(this->parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      assert(numVertices == 0 || numVertices == numVerticesTmp);
      numVertices = numVerticesTmp;
   }
}

void ANNLayer::ioParam_verticesA(enum ParamsIOFlag ioFlag) {
   pvAssert(verticesListInParams);
   int numVerticesA = numVertices;
   this->parent->parameters()->ioParamArray(
         ioFlag, this->getName(), "verticesA", &verticesA, &numVerticesA);
   if (ioFlag == PARAMS_IO_READ) {
      if (numVerticesA == 0) {
         if (this->parent->columnId() == 0) {
            ErrorLog().printf("%s: verticesA cannot be empty\n", getDescription_c());
         }
         MPI_Barrier(this->parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      if (numVertices != 0 && numVerticesA != numVertices) {
         if (this->parent->columnId() == 0) {
            ErrorLog().printf(
                  "%s: verticesV (%d elements) and verticesA (%d elements) must have the same "
                  "lengths.\n",
                  getDescription_c(),
                  numVertices,
                  numVerticesA);
         }
         MPI_Barrier(this->parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      assert(numVertices == 0 || numVertices == numVerticesA);
      numVertices = numVerticesA;
   }
}

void ANNLayer::ioParam_slopeNegInf(enum ParamsIOFlag ioFlag) {
   pvAssert(verticesListInParams);
   parent->parameters()->ioParamValue(
         ioFlag, name, "slopeNegInf", &slopeNegInf, slopeNegInf /*default*/, true /*warnIfAbsent*/);
}

void ANNLayer::ioParam_slopePosInf(enum ParamsIOFlag ioFlag) {
   pvAssert(verticesListInParams);
   parent->parameters()->ioParamValue(
         ioFlag, name, "slopePosInf", &slopePosInf, slopePosInf /*default*/, true /*warnIfAbsent*/);
}

void ANNLayer::ioParam_VThresh(enum ParamsIOFlag ioFlag) {
   pvAssert(!verticesListInParams);
   parent->parameters()->ioParamValue(ioFlag, name, "VThresh", &VThresh, VThresh);
}

void ANNLayer::ioParam_AMin(enum ParamsIOFlag ioFlag) {
   pvAssert(!verticesListInParams);
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "VThresh"));
   parent->parameters()->ioParamValue(
         ioFlag,
         name,
         "AMin",
         &AMin,
         VThresh); // defaults to the value of VThresh, which was read earlier.
}

void ANNLayer::ioParam_AMax(enum ParamsIOFlag ioFlag) {
   pvAssert(!verticesListInParams);
   parent->parameters()->ioParamValue(ioFlag, name, "AMax", &AMax, AMax);
}

void ANNLayer::ioParam_AShift(enum ParamsIOFlag ioFlag) {
   pvAssert(!verticesListInParams);
   parent->parameters()->ioParamValue(ioFlag, name, "AShift", &AShift, AShift);
}

void ANNLayer::ioParam_VWidth(enum ParamsIOFlag ioFlag) {
   pvAssert(!verticesListInParams);
   parent->parameters()->ioParamValue(ioFlag, name, "VWidth", &VWidth, VWidth);
}

int ANNLayer::setVertices() {
   pvAssert(!layerListsVerticesInParams());
   if (VWidth < 0) {
      VThresh += VWidth;
      VWidth = -VWidth;
      if (parent->columnId() == 0) {
         WarnLog().printf(
               "%s: interpreting negative VWidth as setting VThresh=%f and VWidth=%f\n",
               getDescription_c(),
               (double)VThresh,
               (double)VWidth);
      }
   }

   float limfromright = VThresh + VWidth - AShift;
   if (AMax < limfromright)
      limfromright = AMax;

   if (AMin > limfromright) {
      if (parent->columnId() == 0) {
         if (VWidth == 0) {
            WarnLog().printf(
                  "%s: nonmonotonic transfer function, jumping from %f to %f at Vthresh=%f\n",
                  getDescription_c(),
                  (double)AMin,
                  (double)limfromright,
                  (double)VThresh);
         }
         else {
            WarnLog().printf(
                  "%s: nonmonotonic transfer function, changing from %f to %f as V goes from "
                  "VThresh=%f to VThresh+VWidth=%f\n",
                  getDescription_c(),
                  (double)AMin,
                  (double)limfromright,
                  (double)VThresh,
                  (double)(VThresh + VWidth));
         }
      }
   }

   // Initialize slopes to NaN so that we can tell whether they've been initialized.
   slopeNegInf = std::numeric_limits<double>::quiet_NaN();
   slopePosInf = std::numeric_limits<double>::quiet_NaN();
   std::vector<float> vectorV;
   std::vector<float> vectorA;

   slopePosInf = 1.0f;
   if (VThresh <= -(float)0.999 * FLT_MAX) {
      numVertices = 1;
      vectorV.push_back((float)0);
      vectorA.push_back(-AShift);
      slopeNegInf = 1.0f;
   }
   else {
      assert(VWidth >= (float)0);
      if (VWidth == (float)0
          && (float)VThresh - AShift == AMin) { // Should there be a tolerance instead of strict ==?
         numVertices = 1;
         vectorV.push_back(VThresh);
         vectorA.push_back(AMin);
      }
      else {
         numVertices = 2;
         vectorV.push_back(VThresh);
         vectorV.push_back(VThresh + VWidth);
         vectorA.push_back(AMin);
         vectorA.push_back(VThresh + VWidth - AShift);
      }
      slopeNegInf = 0.0f;
   }
   if (AMax < (float)0.999 * FLT_MAX) {
      assert(slopePosInf == 1.0f);
      if (vectorA[numVertices - 1] < AMax) {
         float interval = AMax - vectorA[numVertices - 1];
         vectorV.push_back(vectorV[numVertices - 1] + (float)interval);
         vectorA.push_back(AMax);
         numVertices++;
      }
      else {
         // find the last vertex where A < AMax.
         bool found = false;
         int v;
         for (v = numVertices - 1; v >= 0; v--) {
            if (vectorA[v] < AMax) {
               found = true;
               break;
            }
         }
         if (found) {
            assert(v + 1 < numVertices && vectorA[v] < AMax && vectorA[v + 1] >= AMax);
            float interval = AMax - vectorA[v];
            numVertices    = v + 1;
            vectorA.resize(numVertices);
            vectorV.resize(numVertices);
            vectorV.push_back(vectorV[v] + (float)interval);
            vectorA.push_back(AMax);
            // In principle, there could be a case where a vertex n has A[n]>AMax but A[n-1] and
            // A[n+1] are both < AMax.
            // But with the current ANNLayer parameters, that won't happen.
         }
         else {
            // All vertices have A>=AMax.
            // If slopeNegInf is positive, transfer function should increase from -infinity to AMax,
            // and then stays constant.
            // If slopeNegInf is negative or zero,
            numVertices = 1;
            vectorA.resize(numVertices);
            vectorV.resize(numVertices);
            if (slopeNegInf > 0) {
               float intervalA = vectorA[0] - AMax;
               float intervalV = (float)(intervalA / slopeNegInf);
               vectorV[0]      = vectorV[0] - intervalV;
               vectorA[0]      = AMax;
            }
            else {
               // Everything everywhere is above AMax, so make the transfer function a constant
               // A=AMax.
               vectorA.resize(1);
               vectorV.resize(1);
               vectorV[0]  = (float)0;
               vectorA[0]  = AMax;
               numVertices = 1;
               slopeNegInf = 0;
            }
         }
      }
      slopePosInf = 0.0f;
   }
   // Check for NaN
   assert(slopeNegInf == slopeNegInf && slopePosInf == slopePosInf && numVertices > 0);
   assert(vectorA.size() == numVertices && vectorV.size() == numVertices);
   verticesV = (float *)malloc((size_t)numVertices * sizeof(*verticesV));
   verticesA = (float *)malloc((size_t)numVertices * sizeof(*verticesA));
   if (verticesV == NULL || verticesA == NULL) {
      ErrorLog().printf(
            "%s: unable to allocate memory for vertices:%s\n", getDescription_c(), strerror(errno));
      exit(EXIT_FAILURE);
   }
   memcpy(verticesV, &vectorV[0], numVertices * sizeof(*verticesV));
   memcpy(verticesA, &vectorA[0], numVertices * sizeof(*verticesA));

   return PV_SUCCESS;
}

void ANNLayer::setSlopes() {
   pvAssert(numVertices > 0);
   pvAssert(verticesA != nullptr);
   pvAssert(verticesV != nullptr);
   slopes = (float *)pvMallocError(
         (size_t)(numVertices + 1) * sizeof(*slopes),
         "%s: unable to allocate memory for transfer function slopes: %s\n",
         getDescription_c(),
         strerror(errno));
   slopes[0] = slopeNegInf;
   for (int k = 1; k < numVertices; k++) {
      float V1 = verticesV[k - 1];
      float V2 = verticesV[k];
      if (V1 != V2) {
         slopes[k] = (verticesA[k] - verticesA[k - 1]) / (V2 - V1);
      }
      else {
         slopes[k] = verticesA[k] > verticesA[k - 1]
                           ? std::numeric_limits<float>::infinity()
                           : verticesA[k] < verticesA[k - 1]
                                   ? -std::numeric_limits<float>::infinity()
                                   : std::numeric_limits<float>::quiet_NaN();
      }
   }
   slopes[numVertices] = slopePosInf;
}

int ANNLayer::checkVertices() const {
   int status = PV_SUCCESS;
   for (int v = 1; v < numVertices; v++) {
      if (verticesV[v] < verticesV[v - 1]) {
         status = PV_FAILURE;
         if (this->parent->columnId() == 0) {
            ErrorLog().printf(
                  "%s: vertices %d and %d: V-coordinates decrease from %f to %f.\n",
                  getDescription_c(),
                  v,
                  v + 1,
                  (double)verticesV[v - 1],
                  (double)verticesV[v]);
         }
      }
      if (verticesA[v] < verticesA[v - 1]) {
         if (this->parent->columnId() == 0) {
            WarnLog().printf(
                  "%s: vertices %d and %d: A-coordinates decrease from %f to %f.\n",
                  getDescription_c(),
                  v,
                  v + 1,
                  (double)verticesA[v - 1],
                  (double)verticesA[v]);
         }
      }
   }
   return status;
}

int ANNLayer::resetGSynBuffers(double timef, double dt) {
   return HyPerLayer::resetGSynBuffers(timef, dt);
}

Response::Status ANNLayer::updateState(double time, double dt) {
   const PVLayerLoc *loc = getLayerLoc();
   float *A              = clayer->activity->data;
   float *V              = getV();
   int num_channels      = getNumChannels();
   float *gSynHead       = GSyn == NULL ? NULL : GSyn[0];
   int nx                = loc->nx;
   int ny                = loc->ny;
   int nf                = loc->nf;
   int num_neurons       = nx * ny * nf;
   int nbatch            = loc->nbatch;
   if (layerListsVerticesInParams()) {
      ANNLayer_vertices_update_state(
            nbatch,
            num_neurons,
            nx,
            ny,
            nf,
            loc->halo.lt,
            loc->halo.rt,
            loc->halo.dn,
            loc->halo.up,
            V,
            numVertices,
            verticesV,
            verticesA,
            slopes,
            num_channels,
            gSynHead,
            A);
   }
   else {
      ANNLayer_threshminmax_update_state(
            nbatch,
            num_neurons,
            nx,
            ny,
            nf,
            loc->halo.lt,
            loc->halo.rt,
            loc->halo.dn,
            loc->halo.up,
            V,
            VThresh,
            AMin,
            AMax,
            AShift,
            VWidth,
            num_channels,
            gSynHead,
            A);
   }
   return Response::SUCCESS;
}

int ANNLayer::setActivity() {
   const PVLayerLoc *loc = getLayerLoc();
   int nx                = loc->nx;
   int ny                = loc->ny;
   int nf                = loc->nf;
   PVHalo const *halo    = &loc->halo;
   int num_neurons       = nx * ny * nf;
   int nbatch            = loc->nbatch;
   int status;
   status = setActivity_PtwiseLinearTransferLayer(
         nbatch,
         num_neurons,
         getCLayer()->activity->data,
         getV(),
         nx,
         ny,
         nf,
         halo->lt,
         halo->rt,
         halo->dn,
         halo->up,
         numVertices,
         verticesV,
         verticesA,
         slopes);
   return status;
}

} // end namespace PV

///////////////////////////////////////////////////////
//
// implementation of ANNLayer kernels
//

void ANNLayer_vertices_update_state(
      const int nbatch,
      const int numNeurons,
      const int nx,
      const int ny,
      const int nf,
      const int lt,
      const int rt,
      const int dn,
      const int up,

      float *V,
      int numVertices,
      float *verticesV,
      float *verticesA,
      float *slopes,
      int num_channels,
      float *GSynHead,
      float *activity) {
   updateV_ANNLayer_vertices(
         nbatch,
         numNeurons,
         V,
         num_channels,
         GSynHead,
         activity,
         numVertices,
         verticesV,
         verticesA,
         slopes,
         nx,
         ny,
         nf,
         lt,
         rt,
         dn,
         up);
}

void ANNLayer_threshminmax_update_state(
      const int nbatch,
      const int numNeurons,
      const int nx,
      const int ny,
      const int nf,
      const int lt,
      const int rt,
      const int dn,
      const int up,

      float *V,
      float VThresh,
      float AMin,
      float AMax,
      float AShift,
      float VWidth,
      int num_channels,
      float *GSynHead,
      float *activity) {
   updateV_ANNLayer_threshminmax(
         nbatch,
         numNeurons,
         V,
         num_channels,
         GSynHead,
         activity,
         VThresh,
         AMin,
         AMax,
         AShift,
         VWidth,
         nx,
         ny,
         nf,
         lt,
         rt,
         dn,
         up);
}
