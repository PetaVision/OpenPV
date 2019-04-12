/*
 * ComponentBasedObject.cpp
 *
 *  Created on: Jun 11, 2016
 *      Author: pschultz
 */

#include "ComponentBasedObject.hpp"

namespace PV {

ComponentBasedObject::ComponentBasedObject() {
   initialize_base();
   // Note that initialize() is not called in the constructor.
   // Instead, derived classes should call ComponentBasedObject::initialize in their own
   // constructor.
}

int ComponentBasedObject::initialize_base() { return PV_SUCCESS; }

void ComponentBasedObject::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   BaseObject::initialize(name, params, comm);
   std::string componentTableName = std::string("ObserverTable \"") + name + "\"";
   Subject::initializeTable(componentTableName.c_str());
}

int ComponentBasedObject::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   // Components, like all BaseObject-derived objects, read their params during instantiation.
   // When writing out the params file, ComponentBasedObjects must pass the write message
   // to their components.
   if (ioFlag == PARAMS_IO_WRITE) {
      for (auto *c : *mTable) {
         auto obj = dynamic_cast<BaseObject *>(c);
         if (obj) {
            obj->ioParams(ioFlag, false, false);
         }
      }
   }
   return PV_SUCCESS;
}

ComponentBasedObject::~ComponentBasedObject() {}

} /* namespace PV */
