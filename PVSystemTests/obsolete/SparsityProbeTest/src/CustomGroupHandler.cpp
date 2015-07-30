/*
 * CustomGroupHandler.cpp
 *
 *  Created on: Mar 2, 2015
 *      Author: pschultz
 */

#include "CustomGroupHandler.hpp"
#include "SparsityProbeTest.hpp"

#include <stddef.h>
#include <stdio.h>

namespace PV {

CustomGroupHandler::CustomGroupHandler() {
}

CustomGroupHandler::~CustomGroupHandler() {
}

ParamGroupType CustomGroupHandler::getGroupType(char const * keyword) {
   ParamGroupType result = UnrecognizedGroupType;
   if (!strcmp(keyword, "SparsityProbeTest")) {
      result = ProbeGroupType;
   }
   else {
      result = UnrecognizedGroupType;
   }
   return result;
}

BaseProbe * CustomGroupHandler::createProbe(char const * keyword, char const * name, HyPerCol * hc) {
   int status;
   BaseProbe * addedGroup = NULL;
   bool errorFound = false;
   if( !strcmp(keyword, "SparsityProbeTest") ) {
      addedGroup = new SparsityProbeTest(name, hc);
      if( !addedGroup ) {
         fprintf(stderr, "Group \"%s\": Unable to create probe\n", name);
         errorFound = true;
      }
   }
   return addedGroup;
}

} /* namespace PV */
