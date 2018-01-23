/*
 * Response.cpp
 *
 *  Created on Jan 22, 2018
 *
 *  The possible return values of the notify and response functions
 *  in the observer pattern.
 */

#include "Response.hpp"

namespace PV {

Response &Response::operator+=(Response const &a) {
   mStatus = mStatus + a();
   return *this;
}

Response operator+(Response const &a, Response const &b) {
   return Response(a() + b());  
}

Response::Status operator+(Response::Status const &a, Response::Status const &b) {
   if (a == Response::SUCCESS and b == Response::SUCCESS) {
      return Response::SUCCESS;
   }
   else if (a == Response::POSTPONE and b == Response::POSTPONE) {
      return Response::POSTPONE;
   }
   else { return Response::PARTIAL; }
}

} // namespace PV
