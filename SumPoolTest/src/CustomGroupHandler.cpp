/*
 * CustomGroupHandler.cpp
 *
 *  Created on: Mar 2, 2015
 *      Author: pschultz
 */

#include "CustomGroupHandler.hpp"
#include "SumPoolTestLayer.hpp"
#include "InputLayer.hpp"

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
   else if (!strcmp(keyword, "SumPoolTestLayer")) {
      result = LayerGroupType;
   }
   else if (!strcmp(keyword, "InputLayer")) {
      result = LayerGroupType;
   }
   else {
      result = UnrecognizedGroupType;
   }
   return result;
}

HyPerLayer * CustomGroupHandler::createLayer(char const * keyword, char const * name, HyPerCol * hc) {
   int status;
   HyPerLayer * addedLayer = NULL;
   bool errorFound = false;
   if( !strcmp(keyword, "SumPoolTestLayer") ) {
      addedLayer = new SumPoolTestLayer(name, hc);
   }
   else if( !strcmp(keyword, "InputLayer") ) {
      addedLayer = new InputLayer(name, hc);
   }

   if( !addedLayer ) {
      fprintf(stderr, "Group \"%s\": Unable to create layer\n", name);
      errorFound = true;
   }
   return addedLayer;
}

} /* namespace PV */
