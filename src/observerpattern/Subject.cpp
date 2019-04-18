/*
 * Subject.cpp
 *
 *  Created on: Jul 30, 2016
 *      Author: pschultz
 */

#include "observerpattern/Subject.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include <numeric>

namespace PV {

Response::Status Subject::notify(
      ObserverTable const &table,
      std::vector<std::shared_ptr<BaseMessage const>> messages,
      bool printFlag) {
   Response::Status returnStatus = Response::NO_ACTION;
   auto &objectVector            = table.getObjectVector();
   std::vector<int> numPostponed(messages.size());
   for (auto &obj : objectVector) {
      for (int msgIdx = 0; msgIdx < messages.size(); msgIdx++) {
         auto &msg               = messages[msgIdx];
         Response::Status status = obj->respond(msg);
         returnStatus            = returnStatus + status;

         // If an object postpones, skip any subsequent messages to that object.
         // But continue onto the next object, in case it is what the postponing
         // object is waiting for.
         if (status == Response::POSTPONE) {
            numPostponed[msgIdx]++;
            if (printFlag) {
               InfoLog().printf(
                     "%s postponed on %s.\n",
                     obj->getDescription_c(),
                     msg->getMessageType().c_str());
            }
            break;
         }
      }
   }
   if (printFlag) {
      for (int msgIdx = 0; msgIdx < messages.size(); msgIdx++) {
         int numPostponedThisMsg = numPostponed.at(msgIdx);
         if (numPostponedThisMsg > 0) {
            InfoLog().printf(
                  "%d objects postponed %s\n",
                  numPostponedThisMsg,
                  messages[msgIdx]->getMessageType().c_str());
         }
      }
   }
   return returnStatus;
}

void Subject::notifyLoop(
      ObserverTable const &table,
      std::vector<std::shared_ptr<BaseMessage const>> messages,
      bool printFlag,
      std::string const &description) {
   Response::Status status = Response::PARTIAL;
   while (status == Response::PARTIAL) {
      status = notify(table, messages, printFlag);
   }
   FatalIf(
         status == Response::POSTPONE,
         "At least one object of %s postponed, but no object of %s progressed.\n",
         description.c_str(),
         description.c_str());
   pvAssert(Response::completed(status));
}

} /* namespace PV */
