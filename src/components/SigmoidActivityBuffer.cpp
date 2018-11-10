/*
 * SigmoidActivityBuffer.cpp
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#include "SigmoidActivityBuffer.hpp"

namespace PV {

SigmoidActivityBuffer::SigmoidActivityBuffer(
      char const *name,
      PVParams *params,
      Communicator *comm) {
   initialize(name, params, comm);
}

SigmoidActivityBuffer::~SigmoidActivityBuffer() {}

void SigmoidActivityBuffer::initialize(char const *name, PVParams *params, Communicator *comm) {
   VInputActivityBuffer::initialize(name, params, comm);
}

void SigmoidActivityBuffer::setObjectType() { mObjectType = "SigmoidActivityBuffer"; }

int SigmoidActivityBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = VInputActivityBuffer::ioParamsFillGroup(ioFlag);
   ioParam_Vrest(ioFlag);
   ioParam_VthRest(ioFlag);
   ioParam_InverseFlag(ioFlag);
   ioParam_SigmoidFlag(ioFlag);
   ioParam_SigmoidAlpha(ioFlag);

   return status;
}

void SigmoidActivityBuffer::ioParam_Vrest(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "Vrest", &mVrest, mVrest);
}
void SigmoidActivityBuffer::ioParam_VthRest(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "VthRest", &mVthRest, mVthRest);
}
void SigmoidActivityBuffer::ioParam_InverseFlag(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "InverseFlag", &mInverseFlag, mInverseFlag);
}
void SigmoidActivityBuffer::ioParam_SigmoidFlag(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "SigmoidFlag", &mSigmoidFlag, mSigmoidFlag);
}
void SigmoidActivityBuffer::ioParam_SigmoidAlpha(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "SigmoidAlpha", &mSigmoidAlpha, mSigmoidAlpha);
}

Response::Status SigmoidActivityBuffer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   Response::Status status = VInputActivityBuffer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   if (mCommunicator->globalCommRank() == 0) {
      if (mInverseFlag)
         InfoLog().printf("%s Inverse flag is set\n", getDescription_c());
      if (mSigmoidFlag)
         InfoLog().printf("%s True Sigmoid flag is set\n", getDescription_c());
   }

   if (mSigmoidAlpha < 0.0f || mSigmoidAlpha > 1.0f) {
      if (mCommunicator->commRank() == 0) {
         ErrorLog().printf(
               "%s SigmoidAlpha cannot be negative or greater than 1 (value is %f).\n",
               getDescription_c(),
               (double)mSigmoidAlpha);
      }
      MPI_Barrier(mCommunicator->communicator());
      exit(EXIT_FAILURE);
   }
   return Response::SUCCESS;
}

Response::Status
SigmoidActivityBuffer::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   FatalIf(mInternalState == nullptr, "%s requires an InternalState buffer.\n", getDescription_c());

   int const numExtendedAcrossBatch = getBufferSizeAcrossBatch();
   float *activityData              = getReadWritePointer();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (int kExt = 0; kExt < numExtendedAcrossBatch; kExt++) {
      activityData[kExt] = 0.0f;
   }
   return Response::SUCCESS;
}

void SigmoidActivityBuffer::updateBufferCPU(double simTime, double deltaTime) {
   float *A              = getReadWritePointer();
   float const *V        = mInternalState->getBufferData();
   const PVLayerLoc *loc = getLayerLoc();
   int const nb          = loc->nbatch;
   int const nx          = loc->nx;
   int const ny          = loc->ny;
   int const nf          = loc->nf;
   int const lt          = loc->halo.lt;
   int const rt          = loc->halo.rt;
   int const dn          = loc->halo.dn;
   int const up          = loc->halo.up;
   pvAssert(mInternalState->getBufferSize() == nx * ny * nf);
   pvAssert(getBufferSize() == (nx + lt + rt) * (ny + dn + up) * nf);
   int numNeuronsAcrossBatch = mInternalState->getBufferSizeAcrossBatch();
   float Vth                 = (mVthRest + mVrest) / 2.0f;
   float sigScale            = -logf(1.0f / mSigmoidAlpha - 1.0f) / (Vth - mVrest);
   if (!mSigmoidFlag) {
      sigScale = sigScale / logf(3.0f);
      // If sigmoid_flag is off, A is a piecewise linear function of V, with slope of sigScale/2 at
      // V=Vth, truncated to have minimum value 0 and maximum value 1.
      // The log(3) term makes alpha=1/4 have the slope such that V reaches 0 at Vrest, and V
      // reaches 1 at VthRest.  Without it, that alpha happens at 0.26894...
   }

   if (mSigmoidFlag) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
      for (int k = 0; k < numNeuronsAcrossBatch; k++) {
         int kex        = kIndexExtendedBatch(k, nb, nx, ny, nf, lt, rt, dn, up);
         float activity = 1.0f / (1.0f + std::exp(2.0f * (V[k] - Vth) * sigScale));
         A[kex]         = activity;
      }
   }
   else {
// If SigmoidFlag is off, A is a piecewise linear function of V, with slope of sigScale/2
// at V=Vth, truncated to have minimum value 0 and maximum value 1.
// The log(3) term makes alpha=1/4 have the slope such that V reaches 0 at Vrest, and V
// reaches 1 at VthRest.  Without it, that alpha happens at 0.26894...
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
      for (int k = 0; k < numNeuronsAcrossBatch; k++) {
         int kex        = kIndexExtendedBatch(k, nb, nx, ny, nf, lt, rt, dn, up);
         float activity = 0.5f - (V[k] - Vth) * sigScale / 2.0f;
         activity       = activity < 0.0f ? 0.0f : activity;
         activity       = activity > 1.0f ? 1.0f : activity;
         A[kex]         = activity;
      }
   }
   if (mInverseFlag) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
      for (int k = 0; k < numNeuronsAcrossBatch; k++) {
         int kex = kIndexExtendedBatch(k, nb, nx, ny, nf, lt, rt, dn, up);
         A[kex]  = 1.0f - A[kex];
      }
   }
}

} // namespace PV
