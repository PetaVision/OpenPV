/*
 * Response.cpp
 *
 *  Created on Jan 22, 2018
 *
 *  The possible return values of the notify and response functions
 *  in the observer pattern.
 */

#ifndef RESPONSE_HPP_
#define RESPONSE_HPP_

namespace PV {

class Response {
  public:
   enum Status { SUCCESS, PARTIAL, POSTPONE };

   Response(Status const &a) : mStatus(a) {}
   Response() : mStatus(SUCCESS) {}
   virtual ~Response() {}

   Status operator()() const { return mStatus; }

   Response &operator+=(Response const &a);

  private:
   Status mStatus;
};

Response operator+(Response const &a, Response const &b);

Response::Status operator+(Response::Status const &a, Response::Status const &b);

} // namespace PV

#endif // RESPONSE_HPP_
