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

void * RescaleLayerTestGroupHandler::createObject(char const * keyword, char const * name, HyPerCol * hc) {
   int status;
   void * addedGroup = NULL;
   bool errorFound = false;
   if( !strcmp(keyword, "RescaleLayerTestProbe") ) {
      addedGroup = (void *) new RescaleLayerTestProbe(name, hc);
      if( !addedGroup ) {
         fprintf(stderr, "Group \"%s\": Unable to create probe\n", name);
         errorFound = true;
      }
   }
   return addedGroup;
}

} /* namespace PV */
