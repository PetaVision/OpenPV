/*
 * BaseMessage.hpp
 *
 *  Created on: Jul 21, 2016
 *      Author: pschultz
 */

#ifndef BASEMESSAGE_HPP_
#define BASEMESSAGE_HPP_

#include "utils/Timer.hpp"
#include <vector> // #include <map>
#include <string>

namespace PV {

class BaseMessage {
public:
   virtual ~BaseMessage() {}
   inline std::string const& getMessageType() const { return mMessageType; }
protected:
   inline void setMessageType(std::string const& messageType) { mMessageType.clear(); mMessageType += messageType;}
   inline void setMessageType(char const * messageType) { mMessageType.clear(); mMessageType += messageType;}
private:
   std::string mMessageType="";
};

template <typename T>
class CommunicateInitInfoMessage : public BaseMessage {
public:
   CommunicateInitInfoMessage(std::vector<T> hierarchy) {
      setMessageType("CommunicateInitInfo");
      mHierarchy = hierarchy;
   }
   std::vector<T> mHierarchy;
};

class AllocateDataMessage : public BaseMessage {
public:
   AllocateDataMessage() {
      setMessageType("AllocateDataStructures");
   }
};

class ConnectionUpdateMessage : public BaseMessage {
public:
   ConnectionUpdateMessage(double simTime=0.0, double deltaTime=0.0) {
      setMessageType("ConnectionUpdate");
      mTime = simTime;
      mDeltaT = deltaTime;
   }
   double mTime;
   double mDeltaT; // TODO: this should be the nbatch-sized vector of adaptive timesteps
};

class ConnectionFinalizeUpdateMessage : public BaseMessage {
public:
   ConnectionFinalizeUpdateMessage(double simTime=0.0, double deltaTime=0.0) {
      setMessageType("ConnectionFinalizeUpdate");
      mTime = simTime;
      mDeltaT = deltaTime;
   }
   double mTime;
   double mDeltaT; // TODO: this should be the nbatch-sized vector of adaptive timesteps
};

class ConnectionOutputMessage : public BaseMessage {
public:
   ConnectionOutputMessage(double simTime=0.0) {
      setMessageType("ConnectionOutput");
      mTime = simTime;
   }
   double mTime;
};

class LayerReceiveAndUpdateMessage : public BaseMessage {
public:
   LayerReceiveAndUpdateMessage(int phase=0, Timer * timer=nullptr, bool recvOnGpuFlag=false, bool updateOnGpuFlag=false, double simTime=0.0, double deltaTime=0.0) {
      setMessageType("LayerReceiveAndUpdate");
      mPhase = phase;
      mTimer = timer;
      mRecvOnGpuFlag = recvOnGpuFlag;
      mUpdateOnGpuFlag = updateOnGpuFlag;
      mTime = simTime;
      mDeltaT = deltaTime;
   }
   int mPhase = 0;
   Timer * mTimer = nullptr;
   bool mRecvOnGpuFlag;
   bool mUpdateOnGpuFlag;
   float mTime;
   float mDeltaT; // TODO: this should be the nbatch-sized vector of adaptive timesteps
};

class LayerUpdateStateMessage : public BaseMessage {
public:
   LayerUpdateStateMessage(int phase=0, bool recvOnGpuFlag=false, bool updateOnGpuFlag=false, double simTime=0.0, double deltaTime=0.0) {
      setMessageType("LayerUpdateState");
      mPhase = phase;
      mRecvOnGpuFlag = recvOnGpuFlag;
      mUpdateOnGpuFlag = updateOnGpuFlag;
      mTime = simTime;
      mDeltaT = deltaTime;
   }
   int mPhase;
   bool mRecvOnGpuFlag;
   bool mUpdateOnGpuFlag;
   float mTime;
   float mDeltaT; // TODO: this should be the nbatch-sized vector of adaptive timesteps
};

class LayerCopyFromGpuMessage : public BaseMessage {
public:
   LayerCopyFromGpuMessage(int phase=0, Timer* timer=nullptr) {
      setMessageType("LayerCopyFromGpu");
      mPhase = phase;
      mTimer = timer;
   }
   int mPhase;
   Timer * mTimer;
};

class LayerPublishMessage : public BaseMessage {
public:
   LayerPublishMessage(int phase=0, double simTime=0.0) {
      setMessageType("LayerPublish");
      mPhase = phase;
      mTime = simTime;
   }
   int mPhase;
   double mTime;
};

class LayerOutputStateMessage : public BaseMessage {
public:
   LayerOutputStateMessage(int phase=0, double simTime=0.0) {
      setMessageType("LayerOutputState");
      mPhase = phase;
      mTime = simTime;
   }
   int mPhase;
   double mTime;
};

class LayerCheckNotANumberMessage : public BaseMessage {
public:
   LayerCheckNotANumberMessage(int phase=0) {
      setMessageType("LayerCheckNotANumber");
      mPhase = phase;
   }
   int mPhase;
};

} /* namespace PV */

#endif /* BASEMESSAGE_HPP_ */
