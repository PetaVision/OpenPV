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
   CommunicateInitInfoMessage() {
      initMessageType();
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
   ConnectionUpdateMessage(double simTime, double deltaTime) {
      initMessageType();
      mTime = simTime;
      mDeltaT = deltaTime;
   }
   ConnectionUpdateMessage() {
      initMessageType();
      mTime = 0.0;
      mDeltaT = 0.0;
   }
   double mTime;
   double mDeltaT; // TODO: this should be the nbatch-sized vector of adaptive timesteps
protected:
   void initMessageType() { setMessageType("ConnectionUpdate"); }
};

class ConnectionFinalizeUpdateMessage : public BaseMessage {
public:
   ConnectionFinalizeUpdateMessage(double simTime, double deltaTime) {
      initMessageType();
      mTime = simTime;
      mDeltaT = deltaTime;
   }
   ConnectionFinalizeUpdateMessage() {
      initMessageType();
      mTime = 0.0;
      mDeltaT = 0.0;
   }
   double mTime;
   double mDeltaT; // TODO: this should be the nbatch-sized vector of adaptive timesteps
protected:
   void initMessageType() { setMessageType("ConnectionFinalizeUpdate"); }
};

class ConnectionOutputMessage : public BaseMessage {
public:
   ConnectionOutputMessage(double simTime) {
      initMessageType();
      mTime = simTime;
   }
   ConnectionOutputMessage() {
      initMessageType();
      mTime = 0.0;
   }
   double mTime;
protected:
   void initMessageType() { setMessageType("ConnectionOutput"); }
};

} /* namespace PV */

#endif /* BASEMESSAGE_HPP_ */
