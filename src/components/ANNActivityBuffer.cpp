/*
 * ANNActivityBuffer.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#include "ANNActivityBuffer.hpp"

#undef PV_RUN_ON_GPU
#include "ANNActivityBuffer.kpp"

#include <cassert>

namespace PV {

ANNActivityBuffer::ANNActivityBuffer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

ANNActivityBuffer::~ANNActivityBuffer() {}

void ANNActivityBuffer::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   HyPerActivityBuffer::initialize(paramsIO, comm);
}

void ANNActivityBuffer::setObjectType() { mObjectType = "ANNActivityBuffer"; }

int ANNActivityBuffer::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = HyPerActivityBuffer::ioParamsFillGroup(ioSwitch);

   if (mParamsIO->isPresent("verticesV")) {
      mVerticesListInParams = true;
      ioParam_verticesV(ioSwitch);
      ioParam_verticesA(ioSwitch);
      ioParam_slopeNegInf(ioSwitch);
      ioParam_slopePosInf(ioSwitch);
   }
   else {
      mVerticesListInParams = false;
      ioParam_VThresh(ioSwitch);
      ioParam_AMin(ioSwitch);
      ioParam_AMax(ioSwitch);
      ioParam_AShift(ioSwitch);
      ioParam_VWidth(ioSwitch);
   }

   return status;
}

void ANNActivityBuffer::ioParam_verticesV(ParamsIOSwitch ioSwitch) {
   pvAssert(mVerticesListInParams);
   mParamsIO->ioParam(ioSwitch, "verticesV", &mVerticesV);
   if (ioSwitch == ParamsIOSwitch::Read) {
      if (mVerticesV.empty()) {
         if (this->mCommunicator->commRank() == 0) {
            ErrorLog().printf("%s: verticesV cannot be empty\n", getDescription_c());
         }
         MPI_Barrier(this->mCommunicator->communicator());
         std::exit(EXIT_FAILURE);
      }
      if (!mVerticesA.empty() and !mVerticesV.size() != mVerticesA.size()) {
         if (this->mCommunicator->commRank() == 0) {
            ErrorLog().printf(
                  "%s: verticesV (%zu elements) and verticesA (%zu elements) must have the same "
                  "lengths.\n",
                  getDescription_c(),
                  mVerticesV.size(),
                  mVerticesA.size());
         }
         MPI_Barrier(this->mCommunicator->communicator());
         std::exit(EXIT_FAILURE);
      }
      assert(mNumVertices == 0 or mNumVertices == static_cast<int>(mVerticesV.size()));
      mNumVertices = static_cast<int>(mVerticesV.size());
   }
}

void ANNActivityBuffer::ioParam_verticesA(ParamsIOSwitch ioSwitch) {
   pvAssert(mVerticesListInParams);
   int numVerticesA = mNumVertices;
   mParamsIO->ioParam(ioSwitch, "verticesA", &mVerticesA);
   if (ioSwitch == ParamsIOSwitch::Read) {
      if (mVerticesA.empty()) {
         if (this->mCommunicator->commRank() == 0) {
            ErrorLog().printf("%s: verticesA cannot be empty\n", getDescription_c());
         }
         MPI_Barrier(this->mCommunicator->communicator());
         std::exit(EXIT_FAILURE);
      }
      if (!mVerticesV.empty() and mVerticesA.size() != mVerticesV.size()) {
         if (this->mCommunicator->commRank() == 0) {
            ErrorLog().printf(
                  "%s: verticesV (%zu elements) and verticesA (%zu elements) must have the same "
                  "lengths.\n",
                  getDescription_c(),
                  mVerticesV.size(),
                  mVerticesA.size());
         }
         MPI_Barrier(this->mCommunicator->communicator());
         std::exit(EXIT_FAILURE);
      }
      assert(mNumVertices == 0 || mNumVertices == numVerticesA);
      mNumVertices = numVerticesA;
   }
}

void ANNActivityBuffer::ioParam_slopeNegInf(ParamsIOSwitch ioSwitch) {
   pvAssert(mVerticesListInParams);
   mParamsIO->ioParam(ioSwitch, "slopeNegInf", &mSlopeNegInf);
}

void ANNActivityBuffer::ioParam_slopePosInf(ParamsIOSwitch ioSwitch) {
   pvAssert(mVerticesListInParams);
   mParamsIO->ioParam(ioSwitch, "slopePosInf", &mSlopePosInf);
}

void ANNActivityBuffer::ioParam_VThresh(ParamsIOSwitch ioSwitch) {
   pvAssert(!mVerticesListInParams);
   mParamsIO->ioParam(ioSwitch, "VThresh", &mVThresh);
}

void ANNActivityBuffer::ioParam_AMin(ParamsIOSwitch ioSwitch) {
   pvAssert(!mVerticesListInParams);
   pvAssert(!mParamsIO->presentAndNotBeenRead("VThresh"));
   switch (ioSwitch) {
      case ParamsIOSwitch::Read:
         if (mParamsIO->isPresent("AMin")) {
            mParamsIO->ioParam(ioSwitch, "AMin", &mAMin);
         }
         else {
            mAMin = mVThresh;
            WarnLog().printf(
                  "Using inferred value %f for parameter %s in group \"%s\"\n",
                  (double)mAMin, "AMin", getName());
         }
         break;
      case ParamsIOSwitch::Write:
         mParamsIO->ioParam(ioSwitch, "AMin", &mAMin);
         break;
      default:
         Fatal().printf("Unrecognized ParamsIOSwitch %d\n", ioSwitch);
         break;
   }
}

void ANNActivityBuffer::ioParam_AMax(ParamsIOSwitch ioSwitch) {
   pvAssert(!mVerticesListInParams);
   mParamsIO->ioParam(ioSwitch, "AMax", &mAMax);
}

void ANNActivityBuffer::ioParam_AShift(ParamsIOSwitch ioSwitch) {
   pvAssert(!mVerticesListInParams);
   mParamsIO->ioParam(ioSwitch, "AShift", &mAShift);
}

void ANNActivityBuffer::ioParam_VWidth(ParamsIOSwitch ioSwitch) {
   pvAssert(!mVerticesListInParams);
   mParamsIO->ioParam(ioSwitch, "VWidth", &mVWidth);
}

Response::Status ANNActivityBuffer::allocateDataStructures() {
   // allocateVertices needs to be called before the base class allocateDataStructures
   // because the base class calls allocateUpdateKernel, which creates CudaBuffers
   // for the vertices and slopes.
   allocateVerticesAndSlopes();

   auto status = HyPerActivityBuffer::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }

   return Response::SUCCESS;
}

void ANNActivityBuffer::allocateVerticesAndSlopes() {
   if (!mVerticesListInParams) {
      pvAssert(mVerticesA.empty() and mVerticesV.empty());
      setVertices();
   }
   checkVertices();
   setSlopes();
}

#ifdef PV_USE_CUDA
void ANNActivityBuffer::allocateUpdateKernel() {
   PVCuda::CudaDevice *device = mCudaDevice;

   size_t size    = mVerticesV.size() * sizeof(mVerticesV[0]);
   mCudaVerticesV = device->createBuffer(size, &getDescription());
   mCudaVerticesA = device->createBuffer(size, &getDescription());
   // One more slope than number of vertices, since there are slopes at +inf and -inf
   mCudaSlopes    = device->createBuffer(size + sizeof(mSlopes_[0]), &getDescription());
}

Response::Status ANNActivityBuffer::copyInitialStateToGPU() {
   Response::Status status = HyPerActivityBuffer::copyInitialStateToGPU();
   if (!Response::completed(status)) {
      return status;
   }
   if (!isUsingGPU()) {
      return status;
   }

   mCudaVerticesV->copyToDevice(mVerticesV.data());
   mCudaVerticesA->copyToDevice(mVerticesA.data());
   mCudaSlopes->copyToDevice(mSlopes_.data());

   return Response::SUCCESS;
}

void ANNActivityBuffer::updateBufferGPU(double simTime, double deltaTime) {
   pvAssert(isUsingGPU()); // or should be in updateBufferCPU() method.
   FatalIf(
         !mInternalState->isUsingGPU(),
         "%s is using CUDA but internal state %s is not.\n",
         getDescription_c(),
         mInternalState->getDescription_c());

   runKernel();
}
#endif // PV_USE_CUDA

void ANNActivityBuffer::setVertices() {
   pvAssert(!mVerticesListInParams);
   if (mVWidth < 0) {
      mVThresh += mVWidth;
      mVWidth = -mVWidth;
      if (mCommunicator->globalCommRank() == 0) {
         WarnLog().printf(
               "%s: interpreting negative VWidth as setting VThresh=%f and VWidth=%f\n",
               getDescription_c(),
               (double)mVThresh,
               (double)mVWidth);
      }
   }

   float limfromright = mVThresh + mVWidth - mAShift;
   if (mAMax < limfromright)
      limfromright = mAMax;

   if (mVThresh > -0.99f * FLT_MAX and mAMin > limfromright) {
      if (mCommunicator->globalCommRank() == 0) {
         if (mVWidth == 0) {
            WarnLog().printf(
                  "%s: nonmonotonic transfer function, jumping from %f to %f at Vthresh=%f\n",
                  getDescription_c(),
                  (double)mAMin,
                  (double)limfromright,
                  (double)mVThresh);
         }
         else {
            WarnLog().printf(
                  "%s: nonmonotonic transfer function, changing from %f to %f as V goes from "
                  "VThresh=%f to VThresh+VWidth=%f\n",
                  getDescription_c(),
                  (double)mAMin,
                  (double)limfromright,
                  (double)mVThresh,
                  (double)(mVThresh + mVWidth));
         }
      }
   }

   // Initialize slopes to NaN so that we can tell whether they've been initialized.
   mSlopeNegInf = std::numeric_limits<double>::quiet_NaN();
   mSlopePosInf = std::numeric_limits<double>::quiet_NaN();

   mSlopePosInf = 1.0f;
   if (mVThresh <= -(float)0.999 * FLT_MAX) {
      mNumVertices = 1;
      mVerticesV.push_back((float)0);
      mVerticesA.push_back(-mAShift);
      mSlopeNegInf = 1.0f;
   }
   else {
      assert(mVWidth >= 0.0f);
      if (mVWidth == 0.0f
          && (float)mVThresh - mAShift
                   == mAMin) { // Should there be a tolerance instead of strict ==?
         mNumVertices = 1;
         mVerticesV.push_back(mVThresh);
         mVerticesA.push_back(mAMin);
      }
      else {
         mNumVertices = 2;
         mVerticesV.push_back(mVThresh);
         mVerticesV.push_back(mVThresh + mVWidth);
         mVerticesA.push_back(mAMin);
         mVerticesA.push_back(mVThresh + mVWidth - mAShift);
      }
      mSlopeNegInf = 0.0f;
   }
   if (mAMax < (float)0.999 * FLT_MAX) {
      assert(mSlopePosInf == 1.0f);
      if (mVerticesA[mNumVertices - 1] < mAMax) {
         float interval = mAMax - mVerticesA[mNumVertices - 1];
         mVerticesV.push_back(mVerticesV[mNumVertices - 1] + (float)interval);
         mVerticesA.push_back(mAMax);
         mNumVertices++;
      }
      else {
         // find the last vertex where A < AMax.
         bool found = false;
         int v;
         for (v = mNumVertices - 1; v >= 0; v--) {
            if (mVerticesA[v] < mAMax) {
               found = true;
               break;
            }
         }
         if (found) {
            assert(v + 1 < mNumVertices && mVerticesA[v] < mAMax && mVerticesA[v + 1] >= mAMax);
            float interval = mAMax - mVerticesA[v];
            mNumVertices   = v + 1;
            mVerticesA.resize(mNumVertices);
            mVerticesV.resize(mNumVertices);
            mVerticesV.push_back(mVerticesV[v] + (float)interval);
            mVerticesA.push_back(mAMax);
            // In principle, there could be a case where a vertex n has A[n]>AMax but A[n-1] and
            // A[n+1] are both < AMax.
            // But with the current ANNLayer parameters, that won't happen.
         }
         else {
            // All vertices have A>=AMax.
            // If slopeNegInf is positive, transfer function should increase from -infinity to AMax,
            // and then stays constant.
            // If slopeNegInf is negative or zero,
            mNumVertices = 1;
            mVerticesA.resize(mNumVertices);
            mVerticesV.resize(mNumVertices);
            if (mSlopeNegInf > 0) {
               float intervalA = mVerticesA[0] - mAMax;
               float intervalV = (float)(intervalA / mSlopeNegInf);
               mVerticesV[0]  = mVerticesV[0] - intervalV;
               mVerticesA[0]      = mAMax;
            }
            else {
               // Everything everywhere is above AMax, so make the transfer function a constant
               // A=AMax.
               mVerticesA.resize(1);
               mVerticesV.resize(1);
               mVerticesV[0] = 0.0f;
               mVerticesA[0]   = mAMax;
               mNumVertices = 1;
               mSlopeNegInf = 0;
            }
         }
      }
      mSlopePosInf = 0.0f;
   }
   // Check for NaN
   assert(mSlopeNegInf == mSlopeNegInf && mSlopePosInf == mSlopePosInf && mNumVertices > 0);
   std::size_t numVertices = (std::size_t)mNumVertices;
   assert(mVerticesA.size() == numVertices && mVerticesV.size() == numVertices);
   mVerticesV = mVerticesV;
   mVerticesA = mVerticesA;
}

void ANNActivityBuffer::setSlopes() {
   pvAssert(mNumVertices > 0);
   pvAssert(!mVerticesA.empty());
   pvAssert(mVerticesV.size() == mVerticesA.size());
   mSlopes_.resize(mNumVertices + 1);
   mSlopes_[0] = mSlopeNegInf;
   for (int k = 1; k < mNumVertices; k++) {
      float V1 = mVerticesV[k - 1];
      float V2 = mVerticesV[k];
      if (V1 != V2) {
         mSlopes_[k] = (mVerticesA[k] - mVerticesA[k - 1]) / (V2 - V1);
      }
      else {
         mSlopes_[k] = mVerticesA[k] > mVerticesA[k - 1]
                            ? std::numeric_limits<float>::infinity()
                            : mVerticesA[k] < mVerticesA[k - 1]
                                    ? -std::numeric_limits<float>::infinity()
                                    : std::numeric_limits<float>::quiet_NaN();
      }
   }
   mSlopes_[mNumVertices] = mSlopePosInf;
}

void ANNActivityBuffer::checkVertices() const {
   int status = PV_SUCCESS;
   for (int v = 1; v < mNumVertices; v++) {
      if (mVerticesV[v] < mVerticesV[v - 1]) {
         status = PV_FAILURE;
         if (this->mCommunicator->globalCommRank() == 0) {
            ErrorLog().printf(
                  "%s: vertices %d and %d: V-coordinates decrease from %f to %f.\n",
                  getDescription_c(),
                  v,
                  v + 1,
                  (double)mVerticesV[v - 1],
                  (double)mVerticesV[v]);
         }
      }
      if (mVerticesA[v] < mVerticesA[v - 1]) {
         if (this->mCommunicator->globalCommRank() == 0) {
            WarnLog().printf(
                  "%s: vertices %d and %d: A-coordinates decrease from %f to %f.\n",
                  getDescription_c(),
                  v,
                  v + 1,
                  (double)mVerticesA[v - 1],
                  (double)mVerticesA[v]);
         }
      }
   }
   FatalIf(
         status != PV_SUCCESS,
         "%s requires V-coordinates to be nondecreasing.\n",
         getDescription_c());
}

void ANNActivityBuffer::updateBufferCPU(double simTime, double deltaTime) {
   const PVLayerLoc *loc = getLayerLoc();
   float *A              = getReadWritePointer();
   float const *V        = mInternalState->getBufferData();
   int numNeurons        = mInternalState->getBufferSize();
   int nbatch            = loc->nbatch;

   if (mVerticesListInParams) {
      applyVerticesANNActivityBufferOnCPU(
            nbatch,
            numNeurons,
            loc->nx,
            loc->ny,
            loc->nf,
            loc->halo.lt,
            loc->halo.rt,
            loc->halo.dn,
            loc->halo.up,
            mNumVertices,
            mVerticesV.data(),
            mVerticesA.data(),
            mSlopes_.data(),
            V,
            A);
   }
   else {
      HyPerActivityBuffer::updateBufferCPU(simTime, deltaTime);
      applyVThresh(
            nbatch,
            numNeurons,
            V,
            mVThresh,
            mAMin,
            mAShift,
            mVWidth,
            A,
            loc->nx,
            loc->ny,
            loc->nf,
            loc->halo.lt,
            loc->halo.rt,
            loc->halo.dn,
            loc->halo.up);
      applyAMax(
            nbatch,
            numNeurons,
            mAMax,
            A,
            loc->nx,
            loc->ny,
            loc->nf,
            loc->halo.lt,
            loc->halo.rt,
            loc->halo.dn,
            loc->halo.up);
   }
}

void ANNActivityBuffer::applyVThresh(
      int nbatch,
      int numNeurons,
      float const *V,
      float VThresh,
      float AMin,
      float AShift,
      float VWidth,
      float *activity,
      int nx,
      int ny,
      int nf,
      int lt,
      int rt,
      int dn,
      int up) {
   if (VThresh > -FLT_MAX) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
      for (int kbatch = 0; kbatch < numNeurons * nbatch; kbatch++) {
         int b                = kbatch / numNeurons;
         int k                = kbatch % numNeurons;
         float const *VBatch  = V + b * numNeurons;
         float *activityBatch = activity + b * (nx + lt + rt) * (ny + up + dn) * nf;
         int kex              = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
         if (VBatch[k] < VThresh) {
            activityBatch[kex] = AMin;
         }
         else if (VBatch[k] < VThresh + VWidth) {
            activityBatch[kex] =
                  AMin + (VThresh + VWidth - AShift - AMin) * (VBatch[k] - VThresh) / VWidth;
         }
         else {
            activityBatch[kex] -= AShift;
         }
      }
   }
}

void ANNActivityBuffer::applyAMax(
      int nbatch,
      int numNeurons,
      float AMax,
      float *activity,
      int nx,
      int ny,
      int nf,
      int lt,
      int rt,
      int dn,
      int up) {
   if (AMax < FLT_MAX) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
      for (int kbatch = 0; kbatch < numNeurons * nbatch; kbatch++) {
         int b                = kbatch / numNeurons;
         int k                = kbatch % numNeurons;
         float *activityBatch = activity + b * (nx + lt + rt) * (ny + up + dn) * nf;
         int kex              = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
         if (activityBatch[kex] > AMax) {
            activityBatch[kex] = AMax;
         }
      }
   }
}

} // namespace PV
