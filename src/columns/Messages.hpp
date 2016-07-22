/*
 * BaseMessage.hpp
 *
 *  Created on: Jul 21, 2016
 *      Author: pschultz
 */

#ifndef BASEMESSAGE_HPP_
#define BASEMESSAGE_HPP_

#include <map>
#include <string>

namespace PV {

struct BaseMessage {
   virtual ~BaseMessage() {} // included so that we can dynamically cast to subclasses
};

template <typename T>
struct CommunicateInitInfoMessage : public BaseMessage {
   std::map<std::string, T> mHierarchy;
};

struct AllocateDataMessage : public BaseMessage {
};

struct ConnectionUpdateMessage : public BaseMessage {
   double mTime = 0.0;
   double mDeltaT = 0.0; // TODO: this should be the nbatch-sized vector of adaptive timesteps
};

struct ConnectionOutputMessage : public BaseMessage {
   double mTime = 0.0;
};

} /* namespace PV */

#endif /* BASEMESSAGE_HPP_ */
