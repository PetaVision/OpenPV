/*
 * BaseMessage.hpp
 *
 *  Created on: Jul 21, 2016
 *      Author: pschultz
 */

#ifndef BASEMESSAGE_HPP_
#define BASEMESSAGE_HPP_

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
      initMessageType();
      mHierarchy = hierarchy;
   }
   std::vector<T> mHierarchy;
protected:
   void initMessageType() { setMessageType("CommunicateInitInfo"); }
};

class AllocateDataMessage : public BaseMessage {
public:
   AllocateDataMessage() {
      initMessageType();
   }
protected:
   void initMessageType() { setMessageType("AllocateDataStructures"); }
};

class ConnectionUpdateMessage : public BaseMessage {
public:
   ConnectionUpdateMessage(double simTime=0.0, double deltaTime=0.0) {
      initMessageType();
      mTime = simTime;
      mDeltaT = deltaTime;
   }
   double mTime;
   double mDeltaT; // TODO: this should be the nbatch-sized vector of adaptive timesteps
protected:
   void initMessageType() { setMessageType("ConnectionUpdate"); }
};

class ConnectionFinalizeUpdateMessage : public BaseMessage {
public:
   ConnectionFinalizeUpdateMessage(double simTime=0.0, double deltaTime=0.0) {
      initMessageType();
      mTime = simTime;
      mDeltaT = deltaTime;
   }
   double mTime;
   double mDeltaT; // TODO: this should be the nbatch-sized vector of adaptive timesteps
protected:
   void initMessageType() { setMessageType("ConnectionFinalizeUpdate"); }
};

class ConnectionOutputMessage : public BaseMessage {
public:
   ConnectionOutputMessage(double simTime=0.0) {
      initMessageType();
      mTime = simTime;
   }
   double mTime;
protected:
   void initMessageType() { setMessageType("ConnectionOutput"); }
};

class LayerPublishMessage : public BaseMessage {
public:
   LayerPublishMessage(int phase=0, double simTime=0.0) {
      initMessageType();
      mPhase = phase;
      mTime = simTime;
   }
   int mPhase;
   double mTime;
protected:
   void initMessageType() { setMessageType("LayerPublish"); }
};

class LayerCheckNotANumberMessage : public BaseMessage {
public:
   LayerCheckNotANumberMessage(int phase=0) {
      initMessageType();
      mPhase = phase;
   }
   int mPhase;
protected:
   void initMessageType() { setMessageType("LayerCheckNotANumber"); }
};

} /* namespace PV */

#endif /* BASEMESSAGE_HPP_ */
