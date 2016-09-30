/*
 * BaseMessage.hpp
 *
 *  Created on: Aug 1, 2016
 *      Author: pschultz
 *
 *  The base class for messages passed using Subject::notify and Observer::respond.
 */

#ifndef BASEMESSAGE_HPP_
#define BASEMESSAGE_HPP_

#include <string>

namespace PV {

class BaseMessage {
  public:
   BaseMessage() {}
   virtual ~BaseMessage() {}
   inline std::string const &getMessageType() const { return mMessageType; }

  protected:
   inline void setMessageType(std::string const &messageType) { mMessageType = messageType; }
   inline void setMessageType(char const *messageType) { mMessageType = messageType; }

  private:
   std::string mMessageType = "";
};

} // namespace PV

#endif /* BASEMESSAGE_HPP_ */
