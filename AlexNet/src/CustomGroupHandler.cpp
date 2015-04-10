/*
 * CustomGroupHandler.cpp
 *
 *  Created on: Mar 2, 2015
 *      Author: pschultz
 */

#include "CustomGroupHandler.hpp"
#include "CIFARGTLayer.hpp"
#include "SoftMaxBackprop.hpp"
#include "ProbeLayer.hpp"
#include "BatchConn.hpp"

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
   else if (!strcmp(keyword, "CIFARGTLayer")) {
      result = LayerGroupType;
   }
   else if (!strcmp(keyword, "SoftMaxBackprop")) {
      result = LayerGroupType;
   }
   else if (!strcmp(keyword, "ProbeLayer")) {
      result = LayerGroupType;
   }
   else if (!strcmp(keyword, "BatchConn")) {
      result = ConnectionGroupType;
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
   if( !strcmp(keyword, "CIFARGTLayer") ) {
      addedLayer = new CIFARGTLayer(name, hc);
   }
   if( !strcmp(keyword, "SoftMaxBackprop") ) {
      addedLayer = new SoftMaxBackprop(name, hc);
   }
   if( !strcmp(keyword, "ProbeLayer") ) {
      addedLayer = new ProbeLayer(name, hc);
   }
   if( !addedLayer ) {
      fprintf(stderr, "Group \"%s\": Unable to create layer\n", name);
      errorFound = true;
   }
   return addedLayer;
}

BaseConnection * CustomGroupHandler::createConnection(char const * keyword, char const * name, PV::HyPerCol * hc, PV::InitWeights * weightInitializer, PV::NormalizeBase * weightNormalizer){
   PV::BaseConnection * connection = NULL;
   if (keyword == NULL || getGroupType(keyword) != PV::ConnectionGroupType) { return connection; }

   else if (!strcmp(keyword, "BatchConn")) { connection = new BatchConn(name, hc); }
   if (connection ==NULL) {
      fprintf(stderr, "Rank %d process unable to create %s \"%s\"\n", hc->columnId(), keyword, name);
      exit(EXIT_FAILURE);
   }
   return connection;
}



} /* namespace PV */
