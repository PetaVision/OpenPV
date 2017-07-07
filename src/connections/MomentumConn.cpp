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

void MomentumConn::ioParam_momentumMethod(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (plasticityFlag) {
      parent->parameters()->ioParamStringRequired(ioFlag, name, "momentumMethod", &momentumMethod);
      if (strcmp(momentumMethod, "simple") == 0) {
         method = SIMPLE;
      }
      else if (strcmp(momentumMethod, "viscosity") == 0) {
         method = VISCOSITY;
      }
      else if (strcmp(momentumMethod, "alex") == 0) {
         method = ALEX;
      }
      else {
         Fatal() << "MomentumConn " << name << ": momentumMethod of " << momentumMethod
                 << " is not known, options are \"simple\", \"viscosity\", and \"alex\"\n";
      }
   }
}

void MomentumConn::ioParam_momentumTau(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (plasticityFlag) {
      pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "momentumMethod"));
      float defaultVal = 0;
      switch (method) {
         case SIMPLE: defaultVal    = 0.25f; break;
         case VISCOSITY: defaultVal = 100.0f; break;
         case ALEX: defaultVal      = 0.9f; break;
         default: pvAssertMessage(0, "Unrecognized momentumMethod\n"); break;
      }

      parent->parameters()->ioParamValue(ioFlag, name, "momentumTau", &momentumTau, defaultVal);
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
            ErrorLog() << getDescription() << ": MomentumConn parameter batchPeriod is obsolete. "
                       << "Instead use the HyPerCol nbatch parameter.\n";
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
   return HyPerConn::updateWeights(arborId);
}

void MomentumConn::applyMomentum(int arbor_ID) {
   // Shared weights done in parallel, parallel in numkernels
   switch (method) {
      case SIMPLE: applyMomentum(arbor_ID, momentumTau, momentumDecay); break;
      case VISCOSITY: applyMomentum(arbor_ID, std::exp(-1.0f / momentumTau), momentumDecay); break;
      case ALEX: applyMomentum(arbor_ID, momentumTau, momentumDecay * getDWMax()); break;
      default: pvAssertMessage(0, "Unrecognized momentumMethod\n"); break;
   }
}

void MomentumConn::applyMomentum(int arbor_ID, float dwFactor, float wFactor) {
   int const numKernels = getNumDataPatches();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
   for (int kernelIdx = 0; kernelIdx < numKernels; kernelIdx++) {
      float *dwdata_start        = get_dwDataHead(arbor_ID, kernelIdx);
      float const *prev_dw_start = get_prev_dwDataHead(arbor_ID, kernelIdx);
      float const *wdata_start   = get_wDataHead(arbor_ID, kernelIdx);
      for (int k = 0; k < nxp * nyp * nfp; k++) {
         dwdata_start[k] += dwFactor * prev_dw_start[k] - wFactor * wdata_start[k];
      }
   }
}

// TODO checkpointing not working with batching, must write checkpoint exactly at period
int MomentumConn::registerData(Checkpointer *checkpointer) {
   int status = HyPerConn::registerData(checkpointer);
   if (plasticityFlag) {
      checkpointWeightPvp(checkpointer, "prev_dW", prev_dwDataStart);
   }
   return status;
}

int MomentumConn::readStateFromCheckpoint(Checkpointer *checkpointer) {
   int status = PV_SUCCESS;
   if (initializeFromCheckpointFlag) {
      status = HyPerConn::readStateFromCheckpoint(checkpointer);
      if (plasticityFlag) {
         checkpointer->readNamedCheckpointEntry(
               std::string(name), std::string("prev_dW"), false /*not constant*/);
      }
   }
   return status;
}

} // end namespace PV
