#include "GaussianNoiseActivityBuffer.hpp"
#include <cstdlib>
#include <random>

namespace PV {

GaussianNoiseActivityBuffer::GaussianNoiseActivityBuffer(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

GaussianNoiseActivityBuffer::~GaussianNoiseActivityBuffer() {
}

void GaussianNoiseActivityBuffer::initialize(char const *name, PVParams *params, Communicator const *comm) {
   HyPerActivityBuffer::initialize(name, params, comm);
}

void GaussianNoiseActivityBuffer::setObjectType() { mObjectType = "GaussianNoiseActivityBuffer"; }

Response::Status
GaussianNoiseActivityBuffer::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
    auto status = HyPerActivityBuffer::initializeState(message);
    mGenerator.seed(rand());
    mDistribution = std::normal_distribution<float>(mMu, mSigma);
    return status;
}

int GaussianNoiseActivityBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerActivityBuffer::ioParamsFillGroup(ioFlag);
   ioParam_mu(ioFlag);
   ioParam_sigma(ioFlag);
   return status;
}

void GaussianNoiseActivityBuffer::ioParam_mu(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "mu", &mMu, mMu);
}

void GaussianNoiseActivityBuffer::ioParam_sigma(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "sigma", &mSigma, mSigma);
}

void GaussianNoiseActivityBuffer::updateBufferCPU(double simTime, double deltaTime) {
   float *A           = mBufferData.data();
   float const *V     = mInternalState->getBufferData();
   int const nbatch   = getLayerLoc()->nbatch;
   int const nx       = getLayerLoc()->nx;
   int const ny       = getLayerLoc()->ny;
   int const nf       = getLayerLoc()->nf;
   PVHalo const *halo = &getLayerLoc()->halo;

   int const numNeuronsAcrossBatch = mInternalState->getBufferSizeAcrossBatch();
   pvAssert(V != nullptr);
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (int k = 0; k < numNeuronsAcrossBatch; k++) {
      int kExt = kIndexExtendedBatch(k, nbatch, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);
      A[kExt]  = V[k] + mDistribution(mGenerator);
   }
}

} // namespace PV
