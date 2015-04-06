/**
 * CustomGroupHandler.cpp
 *
 * Created on: Mar 5, 2015
 *     Author: pschultz
 */

#include "CustomGroupHandler.hpp"
#include "InputLayer.hpp"
#include "GTLayer.hpp"
#include "ComparisonLayer.hpp"

namespace PVMLearning{

PV::ParamGroupType CustomGroupHandler::getGroupType(char const * keyword) {
   PV::ParamGroupType result = PV::UnrecognizedGroupType;
   if (keyword == NULL) { return result; }
   else if (!strcmp(keyword, "InputLayer")) { result = PV::LayerGroupType; }
   else if (!strcmp(keyword, "GTLayer")) { result = PV::LayerGroupType; }
   else if (!strcmp(keyword, "ComparisonLayer")) { result = PV::LayerGroupType; }

   return result;
}

PV::HyPerLayer * CustomGroupHandler::createLayer(char const * keyword, char const * name, PV::HyPerCol * hc) {
   PV::HyPerLayer * layer = NULL;
   if (keyword == NULL || getGroupType(keyword) != PV::LayerGroupType) { return layer; }
   else if (!strcmp(keyword, "InputLayer")) { layer = new InputLayer(name, hc); }
   else if (!strcmp(keyword, "GTLayer")) { layer = new GTLayer(name, hc); }
   else if (!strcmp(keyword, "ComparisonLayer")) { layer = new ComparisonLayer(name, hc); }
   
   if (layer==NULL) {
      fprintf(stderr, "Rank %d process unable to create %s \"%s\"\n", hc->columnId(), keyword, name);
      exit(EXIT_FAILURE);
   }
   return layer;
}

}  // namespace PVMatchingPursuit
