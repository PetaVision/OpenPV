/*
 * ArborSystemTestGroupHandler.cpp
 *
 *  Created on: Dec 30, 2014
 *      Author: pschultz
 */

#include "ArborSystemTestGroupHandler.hpp"
#include "ArborTestProbe.hpp"
#include "ArborTestForOnesProbe.hpp"

#include <stddef.h>
#include <stdio.h>

namespace PV {

ArborSystemTestGroupHandler::ArborSystemTestGroupHandler() {
}

ArborSystemTestGroupHandler::~ArborSystemTestGroupHandler() {
}

ParamGroupType ArborSystemTestGroupHandler::getGroupType(char const * keyword) {
   ParamGroupType result = UnrecognizedGroupType;
   if (!strcmp(keyword, "ArborTestProbe")) {
      result = ProbeGroupType;
   }
   else if (!strcmp(keyword, "ArborTestForOnesProbe")) {
      result = ProbeGroupType;
   }
   else {
      result = UnrecognizedGroupType;
   }
   return result;
}

BaseProbe * ArborSystemTestGroupHandler::createProbe(char const * keyword, char const * name, HyPerCol * hc) {
   int status;
   BaseProbe * addedGroup = NULL;
   bool errorFound = false;
   if (keyword==NULL) {
      return NULL;
   }
   else if (!strcmp(keyword, "ArborTestProbe")) {
      addedGroup = new ArborTestProbe(name, hc);
      if (addedGroup==NULL) {errorFound=true;}
   }
   else if (!strcmp(keyword, "ArborTestForOnesProbe")) {
      addedGroup = new ArborTestForOnesProbe(name, hc);
      if (addedGroup==NULL) {errorFound=true;}
   }
   else {
      fprintf(stderr, "ArborSystemTestGroupHandler: %s is not a recognized keyword.\n", keyword);
   }
   if (errorFound) {
      assert(addedGroup==NULL);
      fprintf(stderr, "%s \"%s\": ArborSystemTestGroupHandler unable to create probe.\n", keyword, name);
      return NULL;
   }
   return addedGroup;
}

} /* namespace PV */
