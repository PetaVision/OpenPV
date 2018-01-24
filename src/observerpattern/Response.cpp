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

Response &Response::operator+=(Response const &a) {
   mStatus = mStatus + a();
   return *this;
}

Response operator+(Response const &a, Response const &b) { return Response(a() + b()); }

Response::Status operator+(Response::Status const &a, Response::Status const &b) {
   if (a == Response::SUCCESS and b == Response::SUCCESS) {
      return Response::SUCCESS;
   }
   else if (a == Response::POSTPONE and b == Response::POSTPONE) {
      return Response::POSTPONE;
   }
   else {
      return Response::PARTIAL;
   }
}

int Response::convertStatusToInt(Response::Status status) {
   int intValue;
   switch (status) {
      case Response::SUCCESS: intValue  = PV_SUCCESS; break;
      case Response::PARTIAL: intValue  = PV_PARTIAL; break;
      case Response::POSTPONE: intValue = PV_POSTPONE; break;
      default: pvAssert(0); break;
   }
   return intValue;
}

Response::Status Response::convertIntToStatus(int deprecatedStatusCode) {
   Response::Status status;
   switch (deprecatedStatusCode) {
      case PV_SUCCESS: status = Response::SUCCESS; break;
      case PV_FAILURE:
         Fatal().printf("Response::convertIntToStatus received failure code.\n");
         break;
      case PV_PARTIAL: status  = Response::PARTIAL; break;
      case PV_POSTPONE: status = Response::POSTPONE; break;
      default:
         Fatal().printf("Unable to convert %d to Response::Status type.\n", deprecatedStatusCode);
         break;
   }
   return status;
}

} // namespace PV
