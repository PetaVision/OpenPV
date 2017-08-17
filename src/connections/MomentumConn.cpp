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
   delete mPreviousDeltaWeights;
}

int MomentumConn::initialize_base() {
   momentumTau    = .25;
   momentumMethod = NULL;
   momentumDecay  = 0;
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

void MomentumConn::allocateWeights() {
   HyPerConn::allocateWeights();
   if (plasticityFlag) {
      mPreviousDeltaWeights = new Weights(name, getWeights());
      mPreviousDeltaWeights->allocateDataStructures();
   }
}

int MomentumConn::updateWeights(int arborId) {
   // Add momentum right before updateWeights
   applyMomentum(arborId);

   // Saved to prevweights
   pvAssert(mPreviousDeltaWeights->getData(arborId));
   std::memcpy(
         mPreviousDeltaWeights->getData(arborId),
         getDeltaWeightsDataStart(arborId),
         sizeof(float) * (std::size_t)(nxp * nyp * nfp * getNumDataPatches()));

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
      float *deltaWeights           = getDeltaWeightsDataHead(arbor_ID, kernelIdx);
      float const *prevDeltaWeights = getPreviousDeltaWeightsDataHead(arbor_ID, kernelIdx);
      float const *weights          = getWeightsDataHead(arbor_ID, kernelIdx);
      for (int k = 0; k < nxp * nyp * nfp; k++) {
         deltaWeights[k] += dwFactor * prevDeltaWeights[k] - wFactor * weights[k];
      }
   }
}

// TODO checkpointing not working with batching, must write checkpoint exactly at period
int MomentumConn::registerData(Checkpointer *checkpointer) {
   int status = HyPerConn::registerData(checkpointer);
   if (plasticityFlag) {
      checkpointWeightPvp(checkpointer, "prev_dW", mPreviousDeltaWeights);
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
