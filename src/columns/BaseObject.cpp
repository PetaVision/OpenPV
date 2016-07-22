/*
 * BaseObject.cpp
 *
 *  Created on: Jan 20, 2016
 *      Author: pschultz
 */

#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include "BaseObject.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

BaseObject::BaseObject() {
   initialize_base();
   // Note that initialize() is not called in the constructor.
   // Instead, derived classes should call BaseObject::initialize in their own
   // constructor.
}

int BaseObject::initialize_base() {
   name = NULL;
   parent = NULL;
   return PV_SUCCESS;
}

int BaseObject::initialize(const char * name, HyPerCol * hc) {
   int status = setName(name);
   if (status==PV_SUCCESS) { status = setParent(hc); }
   if (status==PV_SUCCESS) { status = setDescription(); }
   return status;
}

char const * BaseObject::getKeyword() const {
   return getParent()->parameters()->groupKeywordFromName(getName());
}

int BaseObject::setName(char const * name) {
   pvAssert(this->name==NULL);
   int status = PV_SUCCESS;
   this->name = strdup(name);
   if (this->name==NULL) {
      pvErrorNoExit().printf("could not set name \"%s\": %s\n", name, strerror(errno));
      status = PV_FAILURE;
   }
   return status;
}

int BaseObject::setParent(HyPerCol * hc) {
   pvAssert(parent==NULL);
   HyPerCol * parentCol = dynamic_cast<HyPerCol*>(hc);
   int status = parentCol!=NULL ? PV_SUCCESS : PV_FAILURE;
   if (parentCol) {
      parent = parentCol;
   }
   return status;
}

int BaseObject::setDescription() {
   description.clear();
   description.append(getKeyword()).append(" \"").append(getName()).append("\"");
   return PV_SUCCESS;
}

int BaseObject::respond(BaseMessage const * message) {
   int status = PV_SUCCESS; // TODO: convert to enum
   if (message==nullptr) {
      return PV_SUCCESS;
   }
   else if (ConnectionUpdateMessage const * castMessage = dynamic_cast<ConnectionUpdateMessage const*>(message)) {
      status = respondConnectionUpdate(castMessage);
   }
   else if (ConnectionOutputMessage const * castMessage = dynamic_cast<ConnectionOutputMessage const*>(message)) {
      status = respondConnectionOutput(castMessage);
   }
   else {
      pvError() << "Unrecognized message type\n";
      status = PV_FAILURE;
   }
   return status;
}

BaseObject::~BaseObject() {
   free(name);
}

BaseObject * createBasePVObject(char const * name, HyPerCol * hc) {
   pvErrorNoExit().printf("BaseObject should not be instantiated itself, only derived classes of BaseObject.\n");
   return NULL;
}

} /* namespace PV */
