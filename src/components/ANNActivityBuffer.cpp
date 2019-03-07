/*
 * ANNActivityBuffer.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#include "ANNActivityBuffer.hpp"

#undef PV_RUN_ON_GPU
#include "ANNActivityBuffer.kpp"

namespace PV {

ANNActivityBuffer::ANNActivityBuffer(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

ANNActivityBuffer::~ANNActivityBuffer() {
   free(mVerticesV);
   free(mVerticesA);
   free(mSlopes);
}

void ANNActivityBuffer::initialize(char const *name, PVParams *params, Communicator const *comm) {
   HyPerActivityBuffer::initialize(name, params, comm);
}

void ANNActivityBuffer::setObjectType() { mObjectType = "ANNActivityBuffer"; }

int ANNActivityBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerActivityBuffer::ioParamsFillGroup(ioFlag);

   if (parameters()->arrayPresent(name, "verticesV")) {
      mVerticesListInParams = true;
      ioParam_verticesV(ioFlag);
      ioParam_verticesA(ioFlag);
      ioParam_slopeNegInf(ioFlag);
      ioParam_slopePosInf(ioFlag);
   }
   else {
      mVerticesListInParams = false;
      ioParam_VThresh(ioFlag);
      ioParam_AMin(ioFlag);
      ioParam_AMax(ioFlag);
      ioParam_AShift(ioFlag);
      ioParam_VWidth(ioFlag);
   }

   return status;
}

void ANNActivityBuffer::ioParam_verticesV(enum ParamsIOFlag ioFlag) {
   pvAssert(mVerticesListInParams);
   int numVerticesTmp = mNumVertices;
   this->parameters()->ioParamArray(
         ioFlag, this->getName(), "verticesV", &mVerticesV, &numVerticesTmp);
   if (ioFlag == PARAMS_IO_READ) {
      if (numVerticesTmp == 0) {
         if (this->mCommunicator->commRank() == 0) {
            ErrorLog().printf("%s: verticesV cannot be empty\n", getDescription_c());
         }
         MPI_Barrier(this->mCommunicator->communicator());
         exit(EXIT_FAILURE);
      }
      if (mNumVertices != 0 && numVerticesTmp != mNumVertices) {
         if (this->mCommunicator->commRank() == 0) {
            ErrorLog().printf(
                  "%s: verticesV (%d elements) and verticesA (%d elements) must have the same "
                  "lengths.\n",
                  getDescription_c(),
                  numVerticesTmp,
                  mNumVertices);
         }
         MPI_Barrier(this->mCommunicator->communicator());
         exit(EXIT_FAILURE);
      }
      assert(mNumVertices == 0 || mNumVertices == numVerticesTmp);
      mNumVertices = numVerticesTmp;
   }
}

void ANNActivityBuffer::ioParam_verticesA(enum ParamsIOFlag ioFlag) {
   pvAssert(mVerticesListInParams);
   int numVerticesA = mNumVertices;
   this->parameters()->ioParamArray(
         ioFlag, this->getName(), "verticesA", &mVerticesA, &numVerticesA);
   if (ioFlag == PARAMS_IO_READ) {
      if (numVerticesA == 0) {
         if (this->mCommunicator->commRank() == 0) {
            ErrorLog().printf("%s: verticesA cannot be empty\n", getDescription_c());
         }
         MPI_Barrier(this->mCommunicator->communicator());
         exit(EXIT_FAILURE);
      }
      if (mNumVertices != 0 && numVerticesA != mNumVertices) {
         if (this->mCommunicator->commRank() == 0) {
            ErrorLog().printf(
                  "%s: verticesV (%d elements) and verticesA (%d elements) must have the same "
                  "lengths.\n",
                  getDescription_c(),
                  mNumVertices,
                  numVerticesA);
         }
         MPI_Barrier(this->mCommunicator->communicator());
         exit(EXIT_FAILURE);
      }
      assert(mNumVertices == 0 || mNumVertices == numVerticesA);
      mNumVertices = numVerticesA;
   }
}

void ANNActivityBuffer::ioParam_slopeNegInf(enum ParamsIOFlag ioFlag) {
   pvAssert(mVerticesListInParams);
   parameters()->ioParamValue(
         ioFlag,
         name,
         "slopeNegInf",
         &mSlopeNegInf,
         mSlopeNegInf /*default*/,
         true /*warnIfAbsent*/);
}

void ANNActivityBuffer::ioParam_slopePosInf(enum ParamsIOFlag ioFlag) {
   pvAssert(mVerticesListInParams);
   parameters()->ioParamValue(
         ioFlag,
         name,
         "slopePosInf",
         &mSlopePosInf,
         mSlopePosInf /*default*/,
         true /*warnIfAbsent*/);
}

void ANNActivityBuffer::ioParam_VThresh(enum ParamsIOFlag ioFlag) {
   pvAssert(!mVerticesListInParams);
   parameters()->ioParamValue(ioFlag, name, "VThresh", &mVThresh, mVThresh);
}

void ANNActivityBuffer::ioParam_AMin(enum ParamsIOFlag ioFlag) {
   pvAssert(!mVerticesListInParams);
   pvAssert(!parameters()->presentAndNotBeenRead(name, "VThresh"));
   parameters()->ioParamValue(
         ioFlag,
         name,
         "AMin",
         &mAMin,
         mVThresh); // defaults to the value of VThresh, which was read earlier.
}

void ANNActivityBuffer::ioParam_AMax(enum ParamsIOFlag ioFlag) {
   pvAssert(!mVerticesListInParams);
   parameters()->ioParamValue(ioFlag, name, "AMax", &mAMax, mAMax);
}

void ANNActivityBuffer::ioParam_AShift(enum ParamsIOFlag ioFlag) {
   pvAssert(!mVerticesListInParams);
   parameters()->ioParamValue(ioFlag, name, "AShift", &mAShift, mAShift);
}

void ANNActivityBuffer::ioParam_VWidth(enum ParamsIOFlag ioFlag) {
   pvAssert(!mVerticesListInParams);
   parameters()->ioParamValue(ioFlag, name, "VWidth", &mVWidth, mVWidth);
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
      pvAssert(mVerticesA == nullptr and mVerticesV == nullptr);
      setVertices();
   }
   checkVertices();
   setSlopes();
}

#ifdef PV_USE_CUDA
void ANNActivityBuffer::allocateUpdateKernel() {
   PVCuda::CudaDevice *device = mCudaDevice;

   size_t size    = (std::size_t)mNumVertices * sizeof(*mVerticesV);
   mCudaVerticesV = device->createBuffer(size, &getDescription());
   mCudaVerticesA = device->createBuffer(size, &getDescription());
   mCudaSlopes    = device->createBuffer(size + sizeof(*mSlopes), &getDescription());
}

Response::Status ANNActivityBuffer::copyInitialStateToGPU() {
   Response::Status status = HyPerActivityBuffer::copyInitialStateToGPU();
   if (!Response::completed(status)) {
      return status;
   }
   if (!isUsingGPU()) {
      return status;
   }

   mCudaVerticesV->copyToDevice(mVerticesV);
   mCudaVerticesA->copyToDevice(mVerticesA);
   mCudaSlopes->copyToDevice(mSlopes);

   return Response::SUCCESS;
}

void ANNActivityBuffer::updateBufferGPU(double simTime, double deltaTime) {
   pvAssert(isUsingGPU()); // or should be in updateBufferCPU() method.
   FatalIf(
         !mInternalState->isUsingGPU(),
         "%s is using CUDA but internal state %s is not.\n",
         getDescription(),
         mInternalState->getDescription());

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

   if (mAMin > limfromright) {
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
   std::vector<float> vectorV;
   std::vector<float> vectorA;

   mSlopePosInf = 1.0f;
   if (mVThresh <= -(float)0.999 * FLT_MAX) {
      mNumVertices = 1;
      vectorV.push_back((float)0);
      vectorA.push_back(-mAShift);
      mSlopeNegInf = 1.0f;
   }
   else {
      assert(mVWidth >= (float)0);
      if (mVWidth == (float)0
          && (float)mVThresh - mAShift
                   == mAMin) { // Should there be a tolerance instead of strict ==?
         mNumVertices = 1;
         vectorV.push_back(mVThresh);
         vectorA.push_back(mAMin);
      }
      else {
         mNumVertices = 2;
         vectorV.push_back(mVThresh);
         vectorV.push_back(mVThresh + mVWidth);
         vectorA.push_back(mAMin);
         vectorA.push_back(mVThresh + mVWidth - mAShift);
      }
      mSlopeNegInf = 0.0f;
   }
   if (mAMax < (float)0.999 * FLT_MAX) {
      assert(mSlopePosInf == 1.0f);
      if (vectorA[mNumVertices - 1] < mAMax) {
         float interval = mAMax - vectorA[mNumVertices - 1];
         vectorV.push_back(vectorV[mNumVertices - 1] + (float)interval);
         vectorA.push_back(mAMax);
         mNumVertices++;
      }
      else {
         // find the last vertex where A < AMax.
         bool found = false;
         int v;
         for (v = mNumVertices - 1; v >= 0; v--) {
            if (vectorA[v] < mAMax) {
               found = true;
               break;
            }
         }
         if (found) {
            assert(v + 1 < mNumVertices && vectorA[v] < mAMax && vectorA[v + 1] >= mAMax);
            float interval = mAMax - vectorA[v];
            mNumVertices   = v + 1;
            vectorA.resize(mNumVertices);
            vectorV.resize(mNumVertices);
            vectorV.push_back(vectorV[v] + (float)interval);
            vectorA.push_back(mAMax);
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
            vectorA.resize(mNumVertices);
            vectorV.resize(mNumVertices);
            if (mSlopeNegInf > 0) {
               float intervalA = vectorA[0] - mAMax;
               float intervalV = (float)(intervalA / mSlopeNegInf);
               vectorV[0]      = vectorV[0] - intervalV;
               vectorA[0]      = mAMax;
            }
            else {
               // Everything everywhere is above AMax, so make the transfer function a constant
               // A=AMax.
               vectorA.resize(1);
               vectorV.resize(1);
               vectorV[0]   = (float)0;
               vectorA[0]   = mAMax;
               mNumVertices = 1;
               mSlopeNegInf = 0;
            }
         }
      }
      mSlopePosInf = 0.0f;
   }
   // Check for NaN
   assert(mSlopeNegInf == mSlopeNegInf && mSlopePosInf == mSlopePosInf && mNumVertices > 0);
   assert(vectorA.size() == mNumVertices && vectorV.size() == mNumVertices);
   mVerticesV = (float *)malloc((size_t)mNumVertices * sizeof(*mVerticesV));
   mVerticesA = (float *)malloc((size_t)mNumVertices * sizeof(*mVerticesA));
   if (mVerticesV == NULL || mVerticesA == NULL) {
      ErrorLog().printf(
            "%s: unable to allocate memory for vertices:%s\n", getDescription_c(), strerror(errno));
      exit(EXIT_FAILURE);
   }
   memcpy(mVerticesV, &vectorV[0], mNumVertices * sizeof(*mVerticesV));
   memcpy(mVerticesA, &vectorA[0], mNumVertices * sizeof(*mVerticesA));
}

void ANNActivityBuffer::setSlopes() {
   pvAssert(mNumVertices > 0);
   pvAssert(mVerticesA != nullptr);
   pvAssert(mVerticesV != nullptr);
   mSlopes = (float *)pvMallocError(
         (size_t)(mNumVertices + 1) * sizeof(*mSlopes),
         "%s: unable to allocate memory for transfer function slopes: %s\n",
         getDescription_c(),
         strerror(errno));
   mSlopes[0] = mSlopeNegInf;
   for (int k = 1; k < mNumVertices; k++) {
      float V1 = mVerticesV[k - 1];
      float V2 = mVerticesV[k];
      if (V1 != V2) {
         mSlopes[k] = (mVerticesA[k] - mVerticesA[k - 1]) / (V2 - V1);
      }
      else {
         mSlopes[k] = mVerticesA[k] > mVerticesA[k - 1]
                            ? std::numeric_limits<float>::infinity()
                            : mVerticesA[k] < mVerticesA[k - 1]
                                    ? -std::numeric_limits<float>::infinity()
                                    : std::numeric_limits<float>::quiet_NaN();
      }
   }
   mSlopes[mNumVertices] = mSlopePosInf;
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
            mVerticesV,
            mVerticesA,
            mSlopes,
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
