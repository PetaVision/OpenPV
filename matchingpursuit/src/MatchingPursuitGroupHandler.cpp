/**
 * MatchingPursuitGroupHandler.cpp
 *
 * Created on: Mar 5, 2015
 *     Author: pschultz
 */

#include "MatchingPursuitGroupHandler.hpp"
#include "MatchingPursuitLayer.hpp"
#include "MatchingPursuitResidual.hpp"

namespace PVMatchingPursuit {

PV::ParamGroupType MatchingPursuitGroupHandler::getGroupType(char const * keyword) {
   PV::ParamGroupType result = PV::UnrecognizedGroupType;
   if (keyword == NULL) { return result; }
   else if (!strcmp(keyword, "MatchingPursuitLayer")) { result = PV::LayerGroupType; }
   else if (!strcmp(keyword, "MatchingPursuitResidual")) { result = PV::LayerGroupType; }

   return result;
}

PV::HyPerLayer * MatchingPursuitGroupHandler::createLayer(char const * keyword, char const * name, PV::HyPerCol * hc) {
   PV::HyPerLayer * layer = NULL;
   if (keyword == NULL || getGroupType(keyword) != PV::LayerGroupType) { return layer; }
   else if (!strcmp(keyword, "MatchingPursuitLayer")) { layer = new MatchingPursuitLayer(name, hc); }
   else if (!strcmp(keyword, "MatchingPursuitResidual")) { layer = new MatchingPursuitResidual(name, hc); }
   
   if (layer==NULL) {
      fprintf(stderr, "Rank %d process unable to create %s \"%s\"\n", hc->columnId(), keyword, name);
      exit(EXIT_FAILURE);
   }
   return layer;
}

}  // namespace PVMatchingPursuit
