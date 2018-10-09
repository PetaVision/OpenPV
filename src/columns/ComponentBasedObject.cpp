/*
 * ComponentBasedObject.cpp
 *
 *  Created on: Jun 11, 2016
 *      Author: pschultz
 */

#include "ComponentBasedObject.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

ComponentBasedObject::ComponentBasedObject() {
   initialize_base();
   // Note that initialize() is not called in the constructor.
   // Instead, derived classes should call ComponentBasedObject::initialize in their own
   // constructor.
}

int ComponentBasedObject::initialize_base() { return PV_SUCCESS; }

int ComponentBasedObject::initialize(const char *name, HyPerCol *hc) {
   int status                     = BaseObject::initialize(name, hc);
   std::string componentTableName = std::string("ObserverTable \"") + name + "\"";
   createComponentTable(componentTableName.c_str());
   readParams();
   return status;
}

ComponentBasedObject::~ComponentBasedObject() {}

} /* namespace PV */
