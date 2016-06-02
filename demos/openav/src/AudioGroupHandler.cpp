#include "AudioGroupHandler.hpp"
#include "AudioInputLayer.hpp"

#include <string.h>

AudioGroupHandler::AudioGroupHandler() {}

AudioGroupHandler::~AudioGroupHandler() {}

PV::ParamGroupType AudioGroupHandler::getGroupType(char const * keyword) {
   PV::ParamGroupType result = PV::UnrecognizedGroupType;
   if (keyword == NULL) { return result; }
   else if (!strcmp(keyword, "AudioInputLayer")) { result = PV::LayerGroupType; }
   return result;
}

PV::HyPerLayer * AudioGroupHandler::createLayer(char const * keyword, char const * name, PV::HyPerCol * hc) {
   PV::HyPerLayer * addedLayer = NULL;
   if (keyword == NULL || getGroupType(keyword) != PV::LayerGroupType) { return addedLayer; }
   else if (!strcmp(keyword, "AudioInputLayer")) { addedLayer = new AudioInputLayer(name, hc); }
   if (addedLayer==NULL)
   {
      fprintf(stderr, "Rank %d process unable to add %s \"%s\"\n", hc->columnId(), keyword, name);
      exit(EXIT_FAILURE);
   }
   return addedLayer;
}

/*PV::BaseProbe * AudioGroupHandler::createProbe(char const * keyword, char const * name, PV::HyPerCol * hc) {
   PV::BaseProbe * addedProbe = NULL;
   if (keyword == NULL || getGroupType(keyword) != PV::ProbeGroupType) { return addedProbe; }
   else if (!strcmp(keyword, "SoundProbe")) {
      addedProbe = new SoundProbe(name, hc);
   }
   return addedProbe;
}*/
