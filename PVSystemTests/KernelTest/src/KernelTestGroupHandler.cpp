/*
 * KernelTestGroupHandler.cpp
 *
 *  Created on: Dec 30, 2014
 *      Author: pschultz
 */

#include "KernelTestGroupHandler.hpp"
#include "KernelTestProbe.hpp"

#include <stddef.h>
#include <stdio.h>

namespace PV {

KernelTestGroupHandler::KernelTestGroupHandler() {
}

KernelTestGroupHandler::~KernelTestGroupHandler() {
}

ParamGroupType KernelTestGroupHandler::getGroupType(char const * keyword) {
   ParamGroupType result = UnrecognizedGroupType;
   if (!strcmp(keyword, "KernelTestProbe")) {
      result = ProbeGroupType;
   }
   else {
      result = UnrecognizedGroupType;
   }
   return result;
}

BaseProbe * KernelTestGroupHandler::createProbe(char const * keyword, char const * name, HyPerCol * hc) {
   int status;
   BaseProbe * addedGroup = NULL;
   bool errorFound = false;
   if( !strcmp(keyword, "KernelTestProbe") ) {
      addedGroup = new KernelTestProbe(name, hc);
      if( !addedGroup ) {
         fprintf(stderr, "Group \"%s\": Unable to create probe\n", name);
         errorFound = true;
      }
   }
   return addedGroup;
}

} /* namespace PV */
