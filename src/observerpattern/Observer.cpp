/*
 * Observer.cpp
 *
 *  Created on: Jul 30, 2016
 *      Author: pschultz
 */

#include "Observer.hpp"

namespace PV {

int Observer::initialize() {
   initMessageActionMap();
   return PV_SUCCESS;
}

Response::Status Observer::respond(std::shared_ptr<BaseMessage const> message) {
   auto result = mMessageActionMap.find(message->getMessageType());
   if (result == mMessageActionMap.end()) {
      return Response::NO_ACTION;
   }
   else {
      auto action = result->second;
      return action(message);
   }
}

} /* namespace PV */
