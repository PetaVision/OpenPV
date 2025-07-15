#include "GaussianNoiseActivityBuffer.hpp"
#include <cstdlib>
#include <random>

namespace PV {

GaussianNoiseActivityBuffer::GaussianNoiseActivityBuffer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

GaussianNoiseActivityBuffer::~GaussianNoiseActivityBuffer() {
}

void GaussianNoiseActivityBuffer::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   HyPerActivityBuffer::initialize(paramsIO, comm);
}

void GaussianNoiseActivityBuffer::setObjectType() { mObjectType = "GaussianNoiseActivityBuffer"; }

Response::Status
GaussianNoiseActivityBuffer::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
    auto status = HyPerActivityBuffer::initializeState(message);
    mGenerator.seed(rand());
    mDistribution = std::normal_distribution<float>(mMu, mSigma);
    return status;
}

int GaussianNoiseActivityBuffer::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = HyPerActivityBuffer::ioParamsFillGroup(ioSwitch);
   ioParam_mu(ioSwitch);
   ioParam_sigma(ioSwitch);
   return status;
}

void GaussianNoiseActivityBuffer::ioParam_mu(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "mu", &mMu);
}

void GaussianNoiseActivityBuffer::ioParam_sigma(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "sigma", &mSigma);
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
