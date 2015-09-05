/*
 * CustomGroupHandler.hpp
 *
 *  Created on: Sep 3, 2015
 *      Author: peteschultz
 */

#include "CustomGroupHandler.hpp"
#include "ResetStateOnTriggerTestProbe.hpp"

#include <stddef.h>
#include <stdio.h>

namespace PV {

CustomGroupHandler::CustomGroupHandler() {
}

CustomGroupHandler::~CustomGroupHandler() {
}

ParamGroupType CustomGroupHandler::getGroupType(char const * keyword) {
   ParamGroupType result = UnrecognizedGroupType;
   if (!strcmp(keyword, "ResetStateOnTriggerTestProbe")) {
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
   if( !strcmp(keyword, "ResetStateOnTriggerTestProbe") ) {
      addedGroup = (BaseProbe *) new ResetStateOnTriggerTestProbe(name, hc);
      if( !addedGroup ) {
         fprintf(stderr, "Group \"%s\": Unable to create probe\n", name);
         errorFound = true;
      }
   }
   return addedGroup;
}

} /* namespace PV */
