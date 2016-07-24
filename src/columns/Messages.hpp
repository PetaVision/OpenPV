/*
 * BaseMessage.hpp
 *
 *  Created on: Jul 21, 2016
 *      Author: pschultz
 */

#ifndef BASEMESSAGE_HPP_
#define BASEMESSAGE_HPP_

#include "utils/Timer.hpp"
#include "cMakeHeader.h"
#include <map>
#include <string>

namespace PV {

class BaseMessage {
public:
   virtual ~BaseMessage() {}
   inline std::string const& getMessageType() const { return mMessageType; }
protected:
   inline void setMessageType(std::string const& messageType) { mMessageType = messageType;}
   inline void setMessageType(char const * messageType) { mMessageType = messageType;}
private:
   std::string mMessageType="";
};

template <typename T>
class CommunicateInitInfoMessage : public BaseMessage {
public:
   CommunicateInitInfoMessage(std::map<std::string, T> const& hierarchy) {
      setMessageType("CommunicateInitInfo");
      mHierarchy = hierarchy;
   }
   std::map<std::string, T> mHierarchy;
};

class AllocateDataMessage : public BaseMessage {
public:
   AllocateDataMessage() {
      setMessageType("AllocateDataStructures");
   }
};

class InitializeStateMessage : public BaseMessage {
public:
   InitializeStateMessage() {
      setMessageType("InitializeState");
   }
};

class ConnectionUpdateMessage : public BaseMessage {
public:
   ConnectionUpdateMessage(double simTime, double deltaTime) {
      setMessageType("ConnectionUpdate");
      mTime = simTime;
      mDeltaT = deltaTime;
   }
   double mTime;
   double mDeltaT; // TODO: this should be the nbatch-sized vector of adaptive timesteps
};

class ConnectionFinalizeUpdateMessage : public BaseMessage {
public:
   ConnectionFinalizeUpdateMessage(double simTime, double deltaTime) {
      setMessageType("ConnectionFinalizeUpdate");
      mTime = simTime;
      mDeltaT = deltaTime;
   }
   double mTime;
   double mDeltaT; // TODO: this should be the nbatch-sized vector of adaptive timesteps
};

class ConnectionOutputMessage : public BaseMessage {
public:
   ConnectionOutputMessage(double simTime) {
      setMessageType("ConnectionOutput");
      mTime = simTime;
   }
   double mTime;
};

class LayerReceiveAndUpdateMessage : public BaseMessage {
public:
   LayerReceiveAndUpdateMessage(int phase, Timer * timer,
#ifdef PV_USE_CUDA
         bool recvOnGpuFlag, bool updateOnGpuFlag,
#endif // PV_USE_CUDA
         double simTime, double deltaTime) {
      setMessageType("LayerReceiveAndUpdate");
      mPhase = phase;
      mTimer = timer;
#ifdef PV_USE_CUDA
      mRecvOnGpuFlag = recvOnGpuFlag;
      mUpdateOnGpuFlag = updateOnGpuFlag;
#endif // PV_USE_CUDA
      mTime = simTime;
      mDeltaT = deltaTime;
   }
   int mPhase = 0;
   Timer * mTimer = nullptr;
#ifdef PV_USE_CUDA
   bool mRecvOnGpuFlag;
   bool mUpdateOnGpuFlag;
#endif // PV_USE_CUDA
   float mTime;
   float mDeltaT; // TODO: this should be the nbatch-sized vector of adaptive timesteps
};

class LayerUpdateStateMessage : public BaseMessage {
public:
   LayerUpdateStateMessage(int phase,
#ifdef PV_USE_CUDA
         bool recvOnGpuFlag, bool updateOnGpuFlag,
#endif // PV_USE_CUDA
         double simTime, double deltaTime) {
      setMessageType("LayerUpdateState");
      mPhase = phase;
#ifdef PV_USE_CUDA
      mRecvOnGpuFlag = recvOnGpuFlag;
      mUpdateOnGpuFlag = updateOnGpuFlag;
#endif // PV_USE_CUDA
      mTime = simTime;
      mDeltaT = deltaTime;
   }
   int mPhase;
#ifdef PV_USE_CUDA
   bool mRecvOnGpuFlag;
   bool mUpdateOnGpuFlag;
#endif // PV_USE_CUDA
   float mTime;
   float mDeltaT; // TODO: this should be the nbatch-sized vector of adaptive timesteps
};

#ifdef PV_USE_CUDA
class LayerCopyFromGpuMessage : public BaseMessage {
public:
   LayerCopyFromGpuMessage(int phase, Timer* timer) {
      setMessageType("LayerCopyFromGpu");
      mPhase = phase;
      mTimer = timer;
   }
   int mPhase;
   Timer * mTimer;
};
#endif // PV_USE_CUDA

class LayerPublishMessage : public BaseMessage {
public:
   LayerPublishMessage(int phase, double simTime) {
      setMessageType("LayerPublish");
      mPhase = phase;
      mTime = simTime;
   }
   int mPhase;
   double mTime;
};

class LayerOutputStateMessage : public BaseMessage {
public:
   LayerOutputStateMessage(int phase, double simTime) {
      setMessageType("LayerOutputState");
      mPhase = phase;
      mTime = simTime;
   }
   int mPhase;
   double mTime;
};

class LayerCheckNotANumberMessage : public BaseMessage {
public:
   LayerCheckNotANumberMessage(int phase) {
      setMessageType("LayerCheckNotANumber");
      mPhase = phase;
   }
   int mPhase;
};

} /* namespace PV */

#endif /* BASEMESSAGE_HPP_ */
