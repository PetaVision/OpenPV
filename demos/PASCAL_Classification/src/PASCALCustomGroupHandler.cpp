/*
 * PASCALCustomGroupHandler.cpp
 *
 *  Created on: Sep 23, 2015
 *      Author: pschultz
 */

#include "PASCALCustomGroupHandler.hpp"
#include "HeatMapProbe.hpp"

PASCALCustomGroupHandler::PASCALCustomGroupHandler() {
}

PV::ParamGroupType PASCALCustomGroupHandler::getGroupType(char const * keyword) {
   PV::ParamGroupType groupType = PV::UnrecognizedGroupType;
   if (keyword!=NULL && !strcmp(keyword, "HeatMapProbe")) { return PV::ProbeGroupType; }
   return groupType;
}

PV::BaseProbe * PASCALCustomGroupHandler::createProbe(char const * keyword, char const * name, PV::HyPerCol * hc) {
   PV::BaseProbe * addedProbe = NULL;
   if (keyword && !strcmp(keyword, "HeatMapProbe")) {
      addedProbe = new HeatMapProbe(name, hc);
   }
   if (addedProbe==NULL && getGroupType(keyword)==PV::ProbeGroupType) {
      fprintf(stderr, "createProbe error: unable to add %s \"%s\"\n", keyword, name);
      exit(EXIT_FAILURE);
   }
   return addedProbe;
}


PASCALCustomGroupHandler::~PASCALCustomGroupHandler() {
}

