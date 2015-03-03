/*
 * CustomGroupHandler.cpp
 *
 *  Created on: Mar 2, 2015
 *      Author: pschultz
 */

#include "CustomGroupHandler.hpp"
#include "ShrunkenPatchTestLayer.hpp"
#include "ShrunkenPatchTestProbe.hpp"

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
   else if (!strcmp(keyword, "ShrunkenPatchTestLayer")) {
      result = LayerGroupType;
   }
   else if (!strcmp(keyword, "ShrunkenPatchTestProbe")) {
      result = ProbeGroupType;
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
   if( !strcmp(keyword, "ShrunkenPatchTestLayer") ) {
      addedLayer = new ShrunkenPatchTestLayer(name, hc);
      if( !addedLayer ) {
         fprintf(stderr, "Group \"%s\": Unable to create layer\n", name);
         errorFound = true;
      }
   }
   return addedLayer;
}

BaseProbe * CustomGroupHandler::createProbe(char const * keyword, char const * name, HyPerCol * hc) {
   int status;
   BaseProbe * addedProbe = NULL;
   bool errorFound = false;
   if( !strcmp(keyword, "ShrunkenPatchTestProbe") ) {
      addedProbe = new ShrunkenPatchTestProbe(name, hc);
      if( !addedProbe ) {
         fprintf(stderr, "Group \"%s\": Unable to create probe\n", name);
         errorFound = true;
      }
   }
   return addedProbe;
}

} /* namespace PV */
