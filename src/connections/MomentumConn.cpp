/*
 * MomentumConn.cpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#include "MomentumConn.hpp"
#include <cstring>

namespace PV {

MomentumConn::MomentumConn() { initialize_base(); }

MomentumConn::MomentumConn(const char *name, HyPerCol *hc) : HyPerConn() {
   initialize_base();
   initialize(name, hc);
}

MomentumConn::~MomentumConn() {
   if (momentumMethod) {
      free(momentumMethod);
   }
   if (prev_dwDataStart) {
      free(prev_dwDataStart[0]);
      free(prev_dwDataStart);
   }
}

int MomentumConn::initialize_base() {
   prev_dwDataStart = NULL;
   momentumTau      = .25;
   momentumMethod   = NULL;
   momentumDecay    = 0;
   return PV_SUCCESS;
}

int MomentumConn::allocateDataStructures() {
   int status = HyPerConn::allocateDataStructures();
   if (status == PV_POSTPONE) {
      return status;
   }
   if (!plasticityFlag)
      return status;
   int sx       = nfp;
   int sy       = sx * nxp;
   int sp       = sy * nyp;
   int nPatches = getNumDataPatches();

   const int numAxons = numberOfAxonalArborLists();

   // Allocate dw buffer for previous dw
   prev_dwDataStart       = (float **)pvCalloc(numAxons, sizeof(float *));
   std::size_t numWeights = (std::size_t)(numAxons * nxp * nyp * nfp) * (std::size_t)nPatches;
   prev_dwDataStart[0]    = (float *)pvCalloc(numWeights, sizeof(float));
   for (int arborId = 0; arborId < numAxons; arborId++) {
      prev_dwDataStart[arborId] = (prev_dwDataStart[0] + sp * nPatches * arborId);
      pvAssert(prev_dwDataStart[arborId] != NULL);
   } // loop over arbors

   return PV_SUCCESS;
}

int MomentumConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerConn::ioParamsFillGroup(ioFlag);
   ioParam_momentumMethod(ioFlag);
   ioParam_momentumTau(ioFlag);
   ioParam_momentumDecay(ioFlag);
   ioParam_batchPeriod(ioFlag);
   return status;
}

void MomentumConn::ioParam_momentumTau(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (plasticityFlag) {
      float defaultVal = 0;
      if (strcmp(momentumMethod, "simple") == 0) {
         defaultVal = .25;
      }
      else if (strcmp(momentumMethod, "viscosity") == 0) {
         defaultVal = 100;
      }
      else if (strcmp(momentumMethod, "alex") == 0) {
         defaultVal = .9;
      }

      parent->parameters()->ioParamValue(ioFlag, name, "momentumTau", &momentumTau, defaultVal);
   }
}

/**
 * @brief momentumMethod: The momentum method to use
 * @details Assuming a = dwMax * pre * post
 * simple: deltaW(t) = a + momentumTau * deltaW(t-1)
 * viscosity: deltaW(t) = (deltaW(t-1) * exp(-1/momentumTau)) + a
 * alex: deltaW(t) = momentumTau * delta(t-1) - momentumDecay * dwMax * w(t) + a
 */
void MomentumConn::ioParam_momentumMethod(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (plasticityFlag) {
      parent->parameters()->ioParamStringRequired(ioFlag, name, "momentumMethod", &momentumMethod);
      if (strcmp(momentumMethod, "simple") != 0 && strcmp(momentumMethod, "viscosity") != 0
          && strcmp(momentumMethod, "alex")) {
         Fatal() << "MomentumConn " << name << ": momentumMethod of " << momentumMethod
                 << " is not known, options are \"simple\", \"viscosity\", and \"alex\"\n";
      }
   }
}

void MomentumConn::ioParam_momentumDecay(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (plasticityFlag) {
      parent->parameters()->ioParamValue(
            ioFlag, name, "momentumDecay", &momentumDecay, momentumDecay);
      if (momentumDecay < 0 || momentumDecay > 1) {
         Fatal() << "MomentumConn " << name
                 << ": momentumDecay must be between 0 and 1 inclusive\n";
      }
   }
}

// batchPeriod parameter was marked obsolete Jan 17, 2017.
void MomentumConn::ioParam_batchPeriod(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (plasticityFlag and parent->parameters()->present(name, "batchPeriod")) {
      int obsoleteBatchPeriod = (int)parent->parameters()->value(name, "batchPeriod");
      if (obsoleteBatchPeriod != 1) {
         if (parent->getCommunicator()->globalCommRank() == 0) {
            ErrorLog() << getDescription() << ": MomentumConn parameter batchPeriod is obsolete. " << "Instead use the HyPerCol nbatch parameter.\n";
         }
         MPI_Barrier(parent->getCommunicator()->globalCommunicator());
         exit(EXIT_FAILURE);
      }
   }
}

int MomentumConn::updateWeights(int arborId) {
   // Add momentum right before updateWeights
   for (int kArbor = 0; kArbor < this->numberOfAxonalArborLists(); kArbor++) {
      applyMomentum(arborId);
   }

   // Saved to prevweights
   pvAssert(prev_dwDataStart);
   std::memcpy(
         *prev_dwDataStart,
         *get_dwDataStart(),
         sizeof(float) * numberOfAxonalArborLists() * nxp * nyp * nfp * getNumDataPatches());

   // add dw to w
   for (int kArbor = 0; kArbor < this->numberOfAxonalArborLists(); kArbor++) {
      float *w_data_start = get_wDataStart(kArbor);
      for (long int k = 0; k < patchStartIndex(getNumDataPatches()); k++) {
         w_data_start[k] += get_dwDataStart(kArbor)[k];
      }
   }
   return PV_BREAK;
}

int MomentumConn::applyMomentum(int arbor_ID) {
   int nExt              = preSynapticLayer()->getNumExtended();
   const PVLayerLoc *loc = preSynapticLayer()->getLayerLoc();
   int numKernels        = getNumDataPatches();

   // Shared weights done in parallel, parallel in numkernels
   if (!strcmp(momentumMethod, "simple")) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for (int kernelIdx = 0; kernelIdx < numKernels; kernelIdx++) {
         float *dwdata_start        = get_dwDataHead(arbor_ID, kernelIdx);
         float const *prev_dw_start = get_prev_dwDataHead(arbor_ID, kernelIdx);
         float const *wdata_start   = get_wDataHead(arbor_ID, kernelIdx);
         for (int k = 0; k < nxp * nyp * nfp; k++) {
            dwdata_start[k] += momentumTau * prev_dw_start[k] - momentumDecay * wdata_start[k];
         }
      }
   }
   else if (!strcmp(momentumMethod, "viscosity")) {
      float const tauFactor = exp(-1.0f / momentumTau);
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for (int kernelIdx = 0; kernelIdx < numKernels; kernelIdx++) {
         float *dwdata_start        = get_dwDataHead(arbor_ID, kernelIdx);
         float const *prev_dw_start = get_prev_dwDataHead(arbor_ID, kernelIdx);
         float const *wdata_start   = get_wDataHead(arbor_ID, kernelIdx);
         for (int k = 0; k < nxp * nyp * nfp; k++) {
            dwdata_start[k] += tauFactor * prev_dw_start[k] - momentumDecay * wdata_start[k];
         }
      }
   }
   else if (!strcmp(momentumMethod, "alex")) {
      float const decayFactor = momentumDecay * getDWMax();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for (int kernelIdx = 0; kernelIdx < numKernels; kernelIdx++) {
         float *dwdata_start        = get_dwDataHead(arbor_ID, kernelIdx);
         float const *prev_dw_start = get_prev_dwDataHead(arbor_ID, kernelIdx);
         float const *wdata_start   = get_wDataHead(arbor_ID, kernelIdx);
         for (int k = 0; k < nxp * nyp * nfp; k++) {
            dwdata_start[k] +=
                  momentumTau * prev_dw_start[k] - decayFactor * wdata_start[k];
         }
      }
   }
   else {
      pvAssertMessage(0, "Unrecognized momentumMethod\n");
   }
   return PV_SUCCESS;
}

// TODO checkpointing not working with batching, must write checkpoint exactly at period
int MomentumConn::registerData(Checkpointer *checkpointer, std::string const &objName) {
   int status = HyPerConn::registerData(checkpointer, objName);
   if (!plasticityFlag) {
      return status;
   }
   checkpointWeightPvp(checkpointer, "prev_dW", prev_dwDataStart);
   return status;
}

int MomentumConn::readStateFromCheckpoint(Checkpointer *checkpointer) {
   int status = PV_SUCCESS;
   if (initializeFromCheckpointFlag) {
      status = HyPerConn::readStateFromCheckpoint(checkpointer);
      checkpointer->readNamedCheckpointEntry(std::string(name), std::string("prev_dW"));
   }
   return status;
}

} // end namespace PV
