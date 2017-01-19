/*
 * Subject.cpp
 *
 *  Created on: Jul 30, 2016
 *      Author: pschultz
 */

#include "observerpattern/Subject.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"

namespace PV {

void Subject::notify(
      ObserverTable const &table,
      std::vector<std::shared_ptr<BaseMessage const>> messages,
      bool printFlag) {
   auto needsUpdate    = table.getObjectVector();
   auto numNeedsUpdate = needsUpdate.size();
   while (numNeedsUpdate > 0) {
      auto oldNumNeedsUpdate = numNeedsUpdate;
      auto iter              = needsUpdate.begin();
      while (iter != needsUpdate.end()) {
         auto obj   = (*iter);
         int status = PV_SUCCESS;
         for (auto &msg : messages) {
            status = obj->respond(msg);
            if (status == PV_BREAK) {
               status = PV_SUCCESS;
            } // Can we get rid of PV_BREAK as a possible return value of
            // connections' updateState?
            switch (status) {
               case PV_SUCCESS: continue; break;
               case PV_POSTPONE: break;
               case PV_FAILURE:
                  Fatal() << obj->getDescription() << " failed " << msg->getMessageType() << ".\n";
                  break;
               default:
                  Fatal() << obj->getDescription() << ": response to " << msg->getMessageType()
                          << " returned unrecognized return code " << status << ".\n";
                  break;
            }
         }
         switch (status) {
            case PV_SUCCESS: iter = needsUpdate.erase(iter); break;
            case PV_POSTPONE: iter++; break;
            default: pvAssert(0); break;
         }
      }
      numNeedsUpdate = needsUpdate.size();
      if (numNeedsUpdate == oldNumNeedsUpdate) {
         if (printFlag) {
            ErrorLog() << "HyPerCol::notify has hung with " << numNeedsUpdate
                       << " objects still postponed.\n";
            for (auto &obj : needsUpdate) {
               ErrorLog() << obj->getDescription() << " is still postponed.\n";
            }
         }
         exit(EXIT_FAILURE);
         break;
      }
      else if (printFlag and numNeedsUpdate > 0) {
         for (auto &msg : messages) {
            if (numNeedsUpdate == 1) {
               InfoLog() << "1 object has still postponed " << msg->getMessageType() << ".\n";
            }
            else {
               InfoLog() << numNeedsUpdate << " objects have still postponed "
                         << msg->getMessageType() << ".\n";
            }
         }
      }
   }
}

} /* namespace PV */
