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

} // namespace Response

} // namespace PV
