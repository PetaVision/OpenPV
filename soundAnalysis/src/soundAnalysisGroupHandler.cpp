/*
 * soundAnalysisGroupHandler.cpp
 *
 *  Created on: Mar 4, 2015
 *      Author: Pete Schultz
 */

#include "soundAnalysisGroupHandler.hpp"
#include "CochlearLayer.hpp"
#include "StreamReconLayer.h"
#include "inverseCochlearLayer.hpp"
#include "inverseNewCochlearLayer.hpp"
#include "SoundProbe.hpp"
#include <string.h>

soundAnalysisGroupHandler::soundAnalysisGroupHandler() {}

soundAnalysisGroupHandler::~soundAnalysisGroupHandler() {}

PV::ParamGroupType soundAnalysisGroupHandler::getGroupType(char const * keyword) {
   PV::ParamGroupType result = PV::UnrecognizedGroupType;
   if (keyword == NULL) { return result; }
   else if (!strcmp(keyword, "CochlearLayer")) {
      result = PV::LayerGroupType;
   }
   else if (!strcmp(keyword, "StreamReconLayer")) {
      result = PV::LayerGroupType;
   }
   else if (!strcmp(keyword, "inverseCochlearLayer")) {
      result = PV::LayerGroupType;
   }
   else if (!strcmp(keyword, "inverseNewCochlearLayer")) {
      result = PV::LayerGroupType;
   }
   else if (!strcmp(keyword, "SoundProbe")) {
      result = PV::LayerGroupType;
   }
   return result;
}

PV::HyPerLayer * soundAnalysisGroupHandler::createLayer(char const * keyword, char const * name, PV::HyPerCol * hc) {
   PV::HyPerLayer * addedLayer = NULL;
   if (keyword == NULL || getGroupType(keyword) != PV::LayerGroupType) { return addedLayer; }
   else if (!strcmp(keyword, "CochlearLayer")) {
      addedLayer = new CochlearLayer(name, hc);
   }
   else if (!strcmp(keyword, "StreamReconLayer")) {
      addedLayer = new StreamReconLayer(name, hc);
   }
   else if (!strcmp(keyword, "inverseCochlearLayer")) {
      addedLayer = new inverseCochlearLayer(name, hc);
   }
   else if (!strcmp(keyword, "inverseNewCochlearLayer")) {
      addedLayer = new inverseNewCochlearLayer(name, hc);
   }
   if (addedLayer==NULL) {
      fprintf(stderr, "Rank %d process unable to add %s \"%s\"\n", hc->columnId(), keyword, name);
      exit(EXIT_FAILURE);
   }
   return addedLayer;
}

PV::BaseProbe * soundAnalysisGroupHandler::createProbe(char const * keyword, char const * name, PV::HyPerCol * hc) {
   PV::BaseProbe * addedProbe = NULL;
   if (keyword == NULL || getGroupType(keyword) != PV::ProbeGroupType) { return addedProbe; }
   else if (!strcmp(keyword, "SoundProbe")) {
      addedProbe = new SoundProbe(name, hc);
   }
   return addedProbe;
}
