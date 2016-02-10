/*
 * CustomGroupHandler.cpp
 *
 *  Created on: Mar 2, 2015
 *      Author: pschultz
 */

#include "CustomGroupHandler.hpp"
#include "AssertZerosProbe.hpp"
#include "SegmentTestLayer.hpp"
#include "SegmentifyTest.hpp"

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
   else if (!strcmp(keyword, "AssertZerosProbe")) {
      result = ProbeGroupType;
   }
   else if (!strcmp(keyword, "SegmentTestLayer")) {
      result = LayerGroupType;
   }
   else if (!strcmp(keyword, "SegmentifyTest")) {
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
   if( !strcmp(keyword, "SegmentTestLayer") ) {
      addedLayer = new SegmentTestLayer(name, hc);
   }
   if( !strcmp(keyword, "SegmentifyTest") ) {
      addedLayer = new SegmentifyTest(name, hc);
   }

   if( !addedLayer ) {
      fprintf(stderr, "Group \"%s\": Unable to create layer\n", name);
      errorFound = true;
   }
   return addedLayer;
}

BaseProbe * CustomGroupHandler::createProbe(char const * keyword, char const * name, HyPerCol * hc) {
   int status;
   BaseProbe * addedProbe = NULL;
   if( !strcmp(keyword, "AssertZerosProbe") ) {
      addedProbe = new AssertZerosProbe(name, hc);
   }
   if( !addedProbe) {
      fprintf(stderr, "Group \"%s\": Unable to create probe\n", name);
   }
   return addedProbe;
}


} /* namespace PV */
