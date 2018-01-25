/*
 * Response.cpp
 *
 *  Created on Jan 22, 2018
 *
 *  The possible return values of the notify and response functions
 *  in the observer pattern.
 */

#include "Response.hpp"
#include "include/pv_common.h"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"

namespace PV {

namespace Response {

Status operator+(Status const &a, Status const &b) {
   if (a == NO_ACTION) {
      return b;
   }
   else if (b == NO_ACTION) {
      return a;
   }
   else if (a == SUCCESS and b == SUCCESS) {
      return SUCCESS;
   }
   else if (a == POSTPONE and b == POSTPONE) {
      return POSTPONE;
   }
   else {
      return PARTIAL;
   }
}

int convertStatusToInt(Response::Status status) {
   int intValue;
   switch (status) {
      case SUCCESS: intValue   = PV_SUCCESS; break;
      case NO_ACTION: intValue = PV_NO_ACTION; break;
      case PARTIAL: intValue   = PV_PARTIAL; break;
      case POSTPONE: intValue  = PV_POSTPONE; break;
      default: pvAssert(0); break;
   }
   return intValue;
}

Status convertIntToStatus(int deprecatedStatusCode) {
   Status status;
   switch (deprecatedStatusCode) {
      case PV_SUCCESS: status = SUCCESS; break;
      case PV_FAILURE:
         Fatal().printf("Response::convertIntToStatus received failure code.\n");
         break;
      case PV_PARTIAL: status   = PARTIAL; break;
      case PV_POSTPONE: status  = POSTPONE; break;
      case PV_NO_ACTION: status = NO_ACTION; break;
      default:
         Fatal().printf("Unable to convert %d to Response::Status type.\n", deprecatedStatusCode);
         break;
   }
   return status;
}

} // namespace Response

} // namespace PV
