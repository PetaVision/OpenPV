/*
 * RescaleLayerTestGroupHandler.cpp
 *
 *  Created on: Dec 30, 2014
 *      Author: pschultz
 */

#include "RescaleLayerTestGroupHandler.hpp"
#include "RescaleLayerTestProbe.hpp"

#include <stddef.h>
#include <stdio.h>

namespace PV {

RescaleLayerTestGroupHandler::RescaleLayerTestGroupHandler() {
}

RescaleLayerTestGroupHandler::~RescaleLayerTestGroupHandler() {
}

ParamGroupType RescaleLayerTestGroupHandler::getGroupType(char const * keyword) {
   ParamGroupType result = UnrecognizedGroupType;
   if (!strcmp(keyword, "RescaleLayerTestProbe")) {
      result = ProbeGroupType;
   }
   else {
      result = UnrecognizedGroupType;
   }
   return result;
}

BaseProbe * RescaleLayerTestGroupHandler::createProbe(char const * keyword, char const * name, HyPerCol * hc) {
   int status;
   BaseProbe * addedProbe = NULL;
   bool errorFound = false;
   if( !strcmp(keyword, "RescaleLayerTestProbe") ) {
      addedProbe = new RescaleLayerTestProbe(name, hc);
      if( !addedProbe ) {
         fprintf(stderr, "Group \"%s\": Unable to create probe\n", name);
         errorFound = true;
      }
   }
   return addedProbe;
}

} /* namespace PV */
