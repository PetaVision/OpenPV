/*
 * BIDSGroupHandler.cpp
 *
 *  Created on: Mar 4, 2015
 *      Author: Pete Schultz
 */

#include "BIDSGroupHandler.hpp"
#include "BIDSLayer.hpp"
#include "BIDSCloneLayer.hpp"
#include "BIDSMovieCloneMap.hpp"
#include "BIDSSensorLayer.hpp"
#include "BIDSConn.hpp"
#include "InitBIDSLateral.hpp"
#include <string.h>

using namespace PV;

namespace PVBIDS {

BIDSGroupHandler::BIDSGroupHandler() {}

BIDSGroupHandler::~BIDSGroupHandler() {}

ParamGroupType BIDSGroupHandler::getGroupType(char const * keyword) {
   ParamGroupType result = UnrecognizedGroupType;
   if (keyword == NULL) { return result; }
   else if (!strcmp(keyword, "BIDSLayer")) {
      result = LayerGroupType;
   }
   else if (!strcmp(keyword, "BIDSCloneLayer")) {
      result = LayerGroupType;
   }
   else if (!strcmp(keyword, "BIDSMovieCloneMap")) {
      result = LayerGroupType;
   }
   else if (!strcmp(keyword, "BIDSSensorLayer")) {
      result = LayerGroupType;
   }
   else if (!strcmp(keyword, "BIDSConn")) {
      result = ConnectionGroupType;
   }
   else if (!strcmp(keyword, "BIDSLateral")) {
      result = WeightInitializerGroupType;
   }
   return result;
}

HyPerLayer * BIDSGroupHandler::createLayer(char const * keyword, char const * name, HyPerCol * hc) {
   HyPerLayer * addedLayer = NULL;
   if (keyword == NULL || getGroupType(keyword) != LayerGroupType) { return addedLayer; }
   else if (!strcmp(keyword, "BIDSLayer")) {
      addedLayer = new BIDSLayer(name, hc);
   }
   else if (!strcmp(keyword, "BIDSCloneLayer")) {
      addedLayer = new BIDSCloneLayer(name, hc);
   }
   else if (!strcmp(keyword, "BIDSMovieCloneMap")) {
      addedLayer = new BIDSMovieCloneMap(name, hc);
   }
   else if (!strcmp(keyword, "BIDSSensorLayer")) {
      addedLayer = new BIDSSensorLayer(name, hc);
   }
   if (addedLayer==NULL) {
      fprintf(stderr, "Rank %d process unable to add %s \"%s\"\n", hc->columnId(), keyword, name);
      exit(EXIT_FAILURE);
   }
   return addedLayer;
}

BaseConnection * BIDSGroupHandler::createConnection(char const * keyword, char const * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer) {
   BaseConnection * addedConn = NULL;
   if (keyword == NULL || getGroupType(keyword) != ConnectionGroupType) { return addedConn; }
   else if (!strcmp(keyword, "BIDSConn")) {
      addedConn = new BIDSConn(name, hc, weightInitializer, weightNormalizer);
   }
   if (addedConn==NULL) {
      fprintf(stderr, "Rank %d process unable to add %s \"%s\"\n", hc->columnId(), keyword, name);
      exit(EXIT_FAILURE);
   }
   return addedConn;
}

InitWeights * BIDSGroupHandler::createWeightInitializer(char const * keyword, char const * name, HyPerCol * hc) {
   InitWeights * addedWeightInitializer = NULL;
   if (keyword == NULL || getGroupType(keyword) != WeightInitializerGroupType) { return addedWeightInitializer; }
   else if (!strcmp(keyword, "BIDSLateral")) {
      addedWeightInitializer = new InitBIDSLateral(name, hc);
   }
   return addedWeightInitializer;
}

}  // namespace PVBIDS
