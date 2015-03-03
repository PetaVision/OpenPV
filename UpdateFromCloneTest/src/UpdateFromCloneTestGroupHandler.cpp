/*
 * UpdateFromCloneTestGroupHandler.cpp
 *
 *  Created on: Mar 2, 2015
 *      Author: pschultz
 */

#include "UpdateFromCloneTestGroupHandler.hpp"
#include "TestConnProbe.hpp"

#include <stddef.h>
#include <stdio.h>

namespace PV {

UpdateFromCloneTestGroupHandler::UpdateFromCloneTestGroupHandler() {
}

UpdateFromCloneTestGroupHandler::~UpdateFromCloneTestGroupHandler() {
}

ParamGroupType UpdateFromCloneTestGroupHandler::getGroupType(char const * keyword) {
   ParamGroupType result = UnrecognizedGroupType;
   if (!strcmp(keyword, "TestConnProbe")) {
      result = ProbeGroupType;
   }
   else {
      result = UnrecognizedGroupType;
   }
   return result;
}

BaseProbe * UpdateFromCloneTestGroupHandler::createProbe(char const * keyword, char const * name, HyPerCol * hc) {
   int status;
   BaseProbe * addedGroup = NULL;
   bool errorFound = false;
   if (keyword==NULL) {
      return addedGroup;
   }
   else if( !strcmp(keyword, "TestConnProbe") ) {
      addedGroup = new TestConnProbe(name, hc);
      if( !addedGroup ) {
         fprintf(stderr, "Group \"%s\": Unable to create probe\n", name);
         errorFound = true;
      }
   }
   return addedGroup;
}

} /* namespace PV */
