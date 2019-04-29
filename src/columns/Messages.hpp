/*
 * Messages.hpp
 *
 *  Created on: Jul 21, 2016
 *      Author: pschultz
 *
 *  The subclasses of BaseMessage used by the HyPerCol.
 */

#ifndef MESSAGES_HPP_
#define MESSAGES_HPP_

#include "cMakeHeader.h"
#include "observerpattern/BaseMessage.hpp"
#include "observerpattern/ObserverTable.hpp"
#include "utils/Timer.hpp"
#include <string>

#ifdef PV_USE_CUDA
#include "arch/cuda/CudaDevice.hpp"
#endif // PV_USE_CUDA

namespace PV {

class CommunicateInitInfoMessage : public BaseMessage {
  public:
   CommunicateInitInfoMessage(
         ObserverTable const *objectTable,
         double deltaTime,
         int nxGlobal,
         int nyGlobal,
         int nBatchGlobal,
         int numThreads) {
      setMessageType("CommunicateInitInfo");
      mObjectTable  = objectTable;
      mDeltaTime    = deltaTime;
      mNxGlobal     = nxGlobal;
      mNyGlobal     = nyGlobal;
      mNBatchGlobal = nBatchGlobal;
      mNumThreads   = numThreads;
   }
   // For lookup, use the ObserverTable function members.
   ObserverTable const *mObjectTable;
   double mDeltaTime;
   int mNxGlobal;
   int mNyGlobal;
   int mNBatchGlobal;
   int mNumThreads;
};

#ifdef PV_USE_CUDA
class SetCudaDeviceMessage : public BaseMessage {
  public:
   SetCudaDeviceMessage(PVCuda::CudaDevice *device) {
      setMessageType("SetCudaDevice");
      mCudaDevice = device;
   }
   PVCuda::CudaDevice *mCudaDevice = nullptr;
};
#endif // PV_USE_CUDA

class AllocateDataStructuresMessage : public BaseMessage {
  public:
   AllocateDataStructuresMessage() { setMessageType("AllocateDataStructures"); }
};

class LayerSetMaxPhaseMessage : public BaseMessage {
  public:
   LayerSetMaxPhaseMessage(int *maxPhase) {
      setMessageType("LayerSetMaxPhase");
      mMaxPhase = maxPhase;
   }
   int *mMaxPhase = nullptr;
};

class LayerWriteParamsMessage : public BaseMessage {
  public:
   LayerWriteParamsMessage() { setMessageType("LayerWriteParams"); }
};

class ConnectionWriteParamsMessage : public BaseMessage {
  public:
   ConnectionWriteParamsMessage() { setMessageType("ConnectionWriteParams"); }
};
class ColProbeWriteParamsMessage : public BaseMessage {
  public:
   ColProbeWriteParamsMessage() { setMessageType("ColProbeWriteParams"); }
};
class LayerProbeWriteParamsMessage : public BaseMessage {
  public:
   LayerProbeWriteParamsMessage() { setMessageType("LayerProbeWriteParams"); }
};
class ConnectionProbeWriteParamsMessage : public BaseMessage {
  public:
   ConnectionProbeWriteParamsMessage() { setMessageType("ConnectionProbeWriteParams"); }
};

class InitializeStateMessage : public BaseMessage {
  public:
   InitializeStateMessage(double deltaTime) {
      setMessageType("InitializeState");
      mDeltaTime = deltaTime;
   }
   double mDeltaTime;
};

class CopyInitialStateToGPUMessage : public BaseMessage {
  public:
   CopyInitialStateToGPUMessage() { setMessageType("CopyInitialStateToGPU"); }
};

class AdaptTimestepMessage : public BaseMessage {
  public:
   AdaptTimestepMessage(double simTime) {
      setMessageType("AdaptTimestep");
      mTime = simTime;
   }
   double mTime;
};

class ConnectionUpdateMessage : public BaseMessage {
  public:
   ConnectionUpdateMessage(double simTime, double deltaTime) {
      setMessageType("ConnectionUpdate");
      mTime   = simTime;
      mDeltaT = deltaTime;
   }
   double mTime;
   double mDeltaT; // TODO: this should be the nbatch-sized vector of adaptive timesteps
};

class ConnectionNormalizeMessage : public BaseMessage {
  public:
   ConnectionNormalizeMessage(double simTime) {
      setMessageType("ConnectionNormalize");
      mTime = simTime;
   }
   double mTime;
};

class ConnectionFinalizeUpdateMessage : public BaseMessage {
  public:
   ConnectionFinalizeUpdateMessage(double simTime, double deltaTime) {
      setMessageType("ConnectionFinalizeUpdate");
      mTime   = simTime;
      mDeltaT = deltaTime;
   }
   double mTime;
   double mDeltaT; // TODO: this should be the nbatch-sized vector of adaptive
   // timesteps
};

class ConnectionOutputMessage : public BaseMessage {
  public:
   ConnectionOutputMessage(double simTime, double deltaTime) {
      setMessageType("ConnectionOutput");
      mTime   = simTime;
      mDeltaT = deltaTime;
   }
   double mTime;
   double mDeltaT;
};

class LayerClearProgressFlagsMessage : public BaseMessage {
  public:
   LayerClearProgressFlagsMessage() { setMessageType("LayerClearProgressFlags"); }
};

class LayerRecvSynapticInputMessage : public BaseMessage {
  public:
   LayerRecvSynapticInputMessage(
         int phase,
         Timer *timer,
#ifdef PV_USE_CUDA
         bool recvOnGpuFlag,
#endif // PV_USE_CUDA
         double simTime,
         double deltaTime,
         bool *someLayerIsPending,
         bool *someLayerHasActed) {
      setMessageType("LayerRecvSynapticInput");
      mPhase = phase;
      mTimer = timer;
#ifdef PV_USE_CUDA
      mRecvOnGpuFlag = recvOnGpuFlag;
#endif // PV_USE_CUDA
      mTime               = simTime;
      mDeltaT             = deltaTime;
      mSomeLayerIsPending = someLayerIsPending;
      mSomeLayerHasActed  = someLayerHasActed;
   }
   int mPhase;
   Timer *mTimer;
#ifdef PV_USE_CUDA
   bool mRecvOnGpuFlag;
#endif // PV_USE_CUDA
   double mTime;
   double mDeltaT; // TODO: this should be the nbatch-sized vector of adaptive
   // timesteps
   bool *mSomeLayerIsPending;
   bool *mSomeLayerHasActed;
};

class LayerUpdateStateMessage : public BaseMessage {
  public:
   LayerUpdateStateMessage(
         int phase,
#ifdef PV_USE_CUDA
         bool recvOnGpuFlag,
         bool updateOnGpuFlag,
// updateState needs recvOnGpuFlag because correct order of updating depends on it.
#endif // PV_USE_CUDA
         double simTime,
         double deltaTime,
         bool *someLayerIsPending,
         bool *someLayerHasActed) {
      setMessageType("LayerUpdateState");
      mPhase = phase;
#ifdef PV_USE_CUDA
      mRecvOnGpuFlag   = recvOnGpuFlag;
      mUpdateOnGpuFlag = updateOnGpuFlag;
#endif // PV_USE_CUDA
      mTime               = simTime;
      mDeltaT             = deltaTime;
      mSomeLayerIsPending = someLayerIsPending;
      mSomeLayerHasActed  = someLayerHasActed;
   }
   int mPhase;
#ifdef PV_USE_CUDA
   bool mRecvOnGpuFlag;
   bool mUpdateOnGpuFlag;
#endif // PV_USE_CUDA
   double mTime;
   double mDeltaT; // TODO: this should be the nbatch-sized vector of adaptive
   // timesteps
   bool *mSomeLayerIsPending;
   bool *mSomeLayerHasActed;
};

#ifdef PV_USE_CUDA
class LayerCopyFromGpuMessage : public BaseMessage {
  public:
   LayerCopyFromGpuMessage(int phase, Timer *timer) {
      setMessageType("LayerCopyFromGpu");
      mPhase = phase;
      mTimer = timer;
   }
   int mPhase;
   Timer *mTimer;
};
#endif // PV_USE_CUDA

class LayerAdvanceDataStoreMessage : public BaseMessage {
  public:
   LayerAdvanceDataStoreMessage(int phase) {
      setMessageType("LayerAdvanceDataStore");
      mPhase = phase;
   }
   int mPhase;
};

class LayerPublishMessage : public BaseMessage {
  public:
   LayerPublishMessage(int phase, double simTime) {
      setMessageType("LayerPublish");
      mPhase = phase;
      mTime  = simTime;
   }
   int mPhase;
   double mTime;
};

class LayerOutputStateMessage : public BaseMessage {
  public:
   LayerOutputStateMessage(int phase, double simTime, double deltaTime) {
      setMessageType("LayerOutputState");
      mPhase     = phase;
      mTime      = simTime;
      mDeltaTime = deltaTime;
   }
   int mPhase;
   double mTime;
   double mDeltaTime;
};

class LayerCheckNotANumberMessage : public BaseMessage {
  public:
   LayerCheckNotANumberMessage(int phase) {
      setMessageType("LayerCheckNotANumber");
      mPhase = phase;
   }
   int mPhase;
};

class ColProbeOutputStateMessage : public BaseMessage {
  public:
   ColProbeOutputStateMessage(double simTime, double deltaTime) {
      setMessageType("ColProbeOutputState");
      mTime      = simTime;
      mDeltaTime = deltaTime;
   }
   double mTime;
   double mDeltaTime;
};

class CleanupMessage : public BaseMessage {
  public:
   CleanupMessage() { setMessageType("Cleanup"); }
};

} /* namespace PV */

#endif /* MESSAGES_HPP_ */
