/**
 * PVsoundGroupHandler.cpp
 *
 * Created on: Mar 5, 2015
 *     Author: pschultz
 */

#include "PVsoundGroupHandler.hpp"
#include "NewCochlear.h"
#include "SoundStream.hpp"

namespace PVsound {

PV::ParamGroupType PVsoundGroupHandler::getGroupType(char const * keyword) {
   PV::ParamGroupType result = PV::UnrecognizedGroupType;
   if (keyword == NULL) { return result; }
   else if (!strcmp(keyword, "NewCochlearLayer")) { result = PV::LayerGroupType; }
   else if (!strcmp(keyword, "SoundStream")) { result = PV::LayerGroupType; }

   return result;
}

PV::HyPerLayer * PVsoundGroupHandler::createLayer(char const * keyword, char const * name, PV::HyPerCol * hc) {
   PV::HyPerLayer * layer = NULL;
   if (keyword == NULL || getGroupType(keyword) != PV::LayerGroupType) { return layer; }
   else if (!strcmp(keyword, "NewCochlearLayer")) { layer = new NewCochlearLayer(name, hc); }
   else if (!strcmp(keyword, "SoundStream")) { layer = new SoundStream(name, hc); }
   
   if (layer==NULL) {
      fprintf(stderr, "Rank %d process unable to create %s \"%s\"\n", hc->columnId(), keyword, name);
      exit(EXIT_FAILURE);
   }
   return layer;
}

}  // namespace PVsound
