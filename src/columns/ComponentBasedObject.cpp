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

void ComponentBasedObject::initialize(const char *name, PVParams *params, Communicator *comm) {
   BaseObject::initialize(name, params, comm);
   std::string componentTableName = std::string("ObserverTable \"") + name + "\"";
   createComponentTable(componentTableName.c_str());
   readParams();
}

ComponentBasedObject::~ComponentBasedObject() {}

} /* namespace PV */
