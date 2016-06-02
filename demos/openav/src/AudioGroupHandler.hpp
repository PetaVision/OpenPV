#pragma once

#include <io/ParamGroupHandler.hpp>

class AudioGroupHandler : public PV::ParamGroupHandler {
public:
   AudioGroupHandler();
   virtual ~AudioGroupHandler();

   virtual PV::ParamGroupType getGroupType(char const * keyword);
   virtual PV::HyPerLayer * createLayer(char const * keyword, char const * name, PV::HyPerCol * hc);
   //virtual PV::BaseProbe * createProbe(char const * keyword, char const * name, PV::HyPerCol * hc);
};
