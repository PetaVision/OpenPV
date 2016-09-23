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

void Subject::notify(ObserverTable const& table, std::vector<std::shared_ptr<BaseMessage const> > messages) {
   auto needsUpdate = table.getObjectVector();
   auto numNeedsUpdate = needsUpdate.size();
   while(numNeedsUpdate>0) {
      auto oldNumNeedsUpdate = numNeedsUpdate;
      auto iter=needsUpdate.begin();
      while (iter!=needsUpdate.end()) {
         auto obj = (*iter);
         int status = PV_SUCCESS;
         for (auto msg : messages) {
            status = obj->respond(msg);
            if (status == PV_BREAK) { status = PV_SUCCESS; } // Can we get rid of PV_BREAK as a possible return value of connections' updateState?
            switch(status) {
            case PV_SUCCESS:
               continue;
               break;
            case PV_POSTPONE:
               pvInfo() << obj->getDescription() << ": " << msg->getMessageType() << " postponed.\n";
               break;
            case PV_FAILURE:
               pvError() << obj->getDescription() << " failed " << msg->getMessageType() << ".\n";
               break;
            default:
               pvError() << obj->getDescription() << " returned unrecognized return code " << status << ".\n";
               break;
            }
         }
         switch(status) {
         case PV_SUCCESS:
            iter = needsUpdate.erase(iter);
            break;
         case PV_POSTPONE:
            iter++;
            break;
         default:
            pvAssert(0);
            break;
         }
      }
      numNeedsUpdate = needsUpdate.size();
      if (numNeedsUpdate == oldNumNeedsUpdate) {
         pvErrorNoExit() << "HyPerCol::notify has hung with " << numNeedsUpdate << " objects still postponed.\n";
         for (auto& obj : needsUpdate) {
            pvErrorNoExit() << obj->getDescription() << " is still postponed.\n";
         }
         exit(EXIT_FAILURE);
         break;
      }
   }
}

} /* namespace PV */
