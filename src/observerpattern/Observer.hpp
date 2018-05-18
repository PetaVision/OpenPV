/*
 * Observer.hpp
 *
 *  Created on: Jul 30, 2016
 *      Author: pschultz
 */

#ifndef OBSERVER_HPP_
#define OBSERVER_HPP_

#include "include/pv_common.h"
#include "observerpattern/BaseMessage.hpp"
#include "observerpattern/Response.hpp"
#include <functional>
#include <map>
#include <memory>

namespace PV {

class Observer {
  public:
   Observer() {}
   virtual ~Observer() {}
   virtual Response::Status respond(std::shared_ptr<BaseMessage const> message) {
      return Response::NO_ACTION;
   }
   inline std::string const &getDescription() const { return mDescription; }
   inline char const *getDescription_c() const { return mDescription.c_str(); }

  protected:
   void setDescription(std::string const &description) { mDescription = description; }
   void setDescription(char const *description) { mDescription = description; }
   int initialize() {
      initMessageActionMap();
      return PV_SUCCESS;
   }

   virtual void initMessageActionMap() {}

   // Data members
  protected:
   std::map<std::string, std::function<Response::Status(std::shared_ptr<BaseMessage const>)>>
         mMessageActionMap;

  private:
   std::string mDescription;
};

} /* namespace PV */

#endif /* OBSERVER_HPP_ */
