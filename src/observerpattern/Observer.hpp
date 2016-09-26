/*
 * Observer.hpp
 *
 *  Created on: Jul 30, 2016
 *      Author: pschultz
 */

#ifndef OBSERVER_HPP_
#define OBSERVER_HPP_

#include "observerpattern/BaseMessage.hpp"
#include "include/pv_common.h"
#include <memory>

namespace PV {

class Observer {
public:
   Observer() {}
   virtual ~Observer() {}
   virtual int respond(std::shared_ptr<BaseMessage const> message) { return PV_SUCCESS; }
   inline std::string const& getDescription() const { return description; }
   inline char const * getDescription_c() const { return description.c_str(); }

// Data members
protected:
   std::string description;
};

} /* namespace PV */

#endif /* OBSERVER_HPP_ */
