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
#include "include/pv_common.h"
#include "columns/HyPerCol.hpp"
#include "utils/PVAssert.hpp"

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
   if (status==PV_SUCCESS) { setParent(hc); }
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
      fprintf(stderr, "Error: could not set name \"%s\": %s\n", name, strerror(errno));
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

BaseObject::~BaseObject() {
   free(name);
}

BaseObject * createBasePVObject(char const * name, HyPerCol * hc) {
   fprintf(stderr, "BaseObject should not be instantiated itself, only derived classes of BaseObject.\n");
   return NULL;
}

} /* namespace PV */
