/**
 * PVtextGroupHandler.cpp
 *
 * Created on: Mar 5, 2015
 *     Author: pschultz
 */

#include "PVtextGroupHandler.hpp"
#include "TextStream.hpp"
#include "TextStreamProbe.hpp"

namespace PVtext {

PV::ParamGroupType PVtextGroupHandler::getGroupType(char const * keyword) {
   PV::ParamGroupType result = PV::UnrecognizedGroupType;
   if (keyword == NULL) { return result; }
   else if (!strcmp(keyword, "TextStream")) { result = PV::LayerGroupType; }
   else if (!strcmp(keyword, "TextStreamProbe")) { result = PV::ProbeGroupType; }

   return result;
}

PV::HyPerLayer * PVtextGroupHandler::createLayer(char const * keyword, char const * name, PV::HyPerCol * hc) {
   PV::HyPerLayer * layer = NULL;
   if (keyword == NULL || getGroupType(keyword) != PV::LayerGroupType) { return layer; }
   else if (!strcmp(keyword, "TextStream")) { layer = new TextStream(name, hc); }
   
   if (layer==NULL) {
      fprintf(stderr, "Rank %d process unable to create %s \"%s\"\n", hc->columnId(), keyword, name);
      exit(EXIT_FAILURE);
   }
   return layer;
}

PV::BaseProbe * PVtextGroupHandler::createProbe(char const * keyword, char const * name, PV::HyPerCol * hc) {
   PV::BaseProbe * probe = NULL;
   if (keyword == NULL || getGroupType(keyword) != PV::ProbeGroupType) { return probe; }
   else if (!strcmp(keyword, "TextStreamProbe")) { probe = new TextStreamProbe(name, hc); }
   
   if (probe==NULL) {
      fprintf(stderr, "Rank %d process unable to create %s \"%s\"\n", hc->columnId(), keyword, name);
      exit(EXIT_FAILURE);
   }
   return probe;
}

}  // namespace PVtext
