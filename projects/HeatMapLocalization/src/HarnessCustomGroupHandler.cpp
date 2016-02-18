/*
 * HarnessCustomGroupHandler.cpp
 *
 *  Created on: Sep 23, 2015
 *      Author: pschultz
 */

#include "HarnessCustomGroupHandler.hpp"
#include "ConvertFromTable.hpp"
#include "LocalizationProbe.hpp"
#include "MaskFromMemoryBuffer.hpp"

HarnessCustomGroupHandler::HarnessCustomGroupHandler() {
}

PV::ParamGroupType HarnessCustomGroupHandler::getGroupType(char const * keyword) {
   PV::ParamGroupType groupType = PV::UnrecognizedGroupType;
   if (keyword!=NULL && !strcmp(keyword, "ConvertFromTable")) { return PV::LayerGroupType; }
   if (keyword!=NULL && !strcmp(keyword, "LocalizationProbe")) { return PV::ProbeGroupType; }
   if (keyword!=NULL && !strcmp(keyword, "MaskFromMemoryBuffer")) { return PV::LayerGroupType; }
   return groupType;
}

PV::HyPerLayer * HarnessCustomGroupHandler::createLayer(char const * keyword, char const * name, PV::HyPerCol * hc) {
   PV::HyPerLayer * addedLayer = NULL;
   if (keyword && !strcmp(keyword, "ConvertFromTable")) {
      addedLayer = new ConvertFromTable(name, hc);
   }
   if (keyword && !strcmp(keyword, "MaskFromMemoryBuffer")) {
      addedLayer = new MaskFromMemoryBuffer(name, hc);
   }
   if (addedLayer==NULL && getGroupType(keyword)==PV::LayerGroupType) {
      fprintf(stderr, "createLayer error: unable to add %s \"%s\"\n", keyword, name);
   }
   return addedLayer;
}

PV::BaseProbe * HarnessCustomGroupHandler::createProbe(char const * keyword, char const * name, PV::HyPerCol * hc) {
   PV::BaseProbe * addedProbe = NULL;
   if (keyword && !strcmp(keyword, "LocalizationProbe")) {
      addedProbe = new LocalizationProbe(name, hc);
   }
   if (addedProbe==NULL && getGroupType(keyword)==PV::ProbeGroupType) {
      fprintf(stderr, "createProbe error: unable to add %s \"%s\"\n", keyword, name);
      exit(EXIT_FAILURE);
   }
   return addedProbe;
}


HarnessCustomGroupHandler::~HarnessCustomGroupHandler() {
}

