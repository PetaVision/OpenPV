/*
 * CustomGroupHandler.cpp
 *
 *  Created on: Mar 2, 2015
 *      Author: pschultz
 */

#include "CustomGroupHandler.hpp"
#include "TriggerTestLayer.hpp"
#include "TriggerTestConn.hpp"
#include "TriggerTestLayerProbe.hpp"

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
   else if (!strcmp(keyword, "TriggerTestLayer")) {
      result = LayerGroupType;
   }
   else if (!strcmp(keyword, "TriggerTestConn")) {
      result = ConnectionGroupType;
   }
   else if (!strcmp(keyword, "TriggerTestLayerProbe")) {
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
   if (keyword==NULL) {
      return addedLayer;
   }
   if( !strcmp(keyword, "TriggerTestLayer") ) {
      addedLayer = new TriggerTestLayer(name, hc);
      if( !addedLayer ) {
         fprintf(stderr, "Group \"%s\": Unable to create layer\n", name);
         errorFound = true;
      }
   }
   return addedLayer;
}

BaseConnection * CustomGroupHandler::createConnection(char const * keyword, char const * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer) {
   int status;
   BaseConnection * addedConn = NULL;
   bool errorFound = false;
   if (keyword==NULL) {
      return addedConn;
   }
   if( !strcmp(keyword, "TriggerTestConn") ) {
      addedConn = new TriggerTestConn(name, hc, weightInitializer, weightNormalizer);
      if( !addedConn ) {
         fprintf(stderr, "Group \"%s\": Unable to create connection\n", name);
         errorFound = true;
      }
   }
   return addedConn;
}

BaseProbe * CustomGroupHandler::createProbe(char const * keyword, char const * name, HyPerCol * hc) {
   int status;
   BaseProbe * addedProbe = NULL;
   bool errorFound = false;
   if (keyword==NULL) {
      return addedProbe;
   }
   if( !strcmp(keyword, "TriggerTestLayerProbe") ) {
      addedProbe = new TriggerTestLayerProbe(name, hc);
      if( !addedProbe ) {
         fprintf(stderr, "Group \"%s\": Unable to create probe\n", name);
         errorFound = true;
      }
   }
   return addedProbe;
}

} /* namespace PV */
