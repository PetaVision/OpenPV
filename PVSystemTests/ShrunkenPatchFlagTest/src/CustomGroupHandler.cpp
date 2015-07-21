/*
 * CustomGroupHandler.cpp
 *
 *  Created on: Mar 2, 2015
 *      Author: pschultz
 */

#include "CustomGroupHandler.hpp"
#include "TestAllZerosProbe.hpp"

#include <stddef.h>
#include <stdio.h>

namespace PV {

CustomGroupHandler::CustomGroupHandler() {
}

CustomGroupHandler::~CustomGroupHandler() {
}

ParamGroupType CustomGroupHandler::getGroupType(char const * keyword) {
   ParamGroupType result = UnrecognizedGroupType;
   if (keyword==NULL) {
      return result;
   }
   else if (!strcmp(keyword, "TestAllZerosProbe")) {
      result = ProbeGroupType;
   }
   else {
      result = UnrecognizedGroupType;
   }
   return result;
}

BaseProbe * CustomGroupHandler::createProbe(char const * keyword, char const * name, HyPerCol * hc) {
   int status;
   BaseProbe * addedProbe = NULL;
   bool errorFound = false;
   if( !strcmp(keyword, "TestAllZerosProbe") ) {
      addedProbe = new TestAllZerosProbe(name, hc);
      if( !addedProbe ) {
         fprintf(stderr, "Group \"%s\": Unable to create probe\n", name);
         errorFound = true;
      }
   }
   return addedProbe;
}

} /* namespace PV */
