/*
 * soundAnalysisGroupHandler.hpp
 *
 *  Created on: Mar 4, 2015
 *      Author: Pete Schultz
 */

#ifndef SOUNDANALYSISGROUPHANDLER_HPP_
#define SOUNDANALYSISGROUPHANDLER_HPP_

#include <io/ParamGroupHandler.hpp>

class soundAnalysisGroupHandler : public PV::ParamGroupHandler {
public:
   soundAnalysisGroupHandler();
   virtual ~soundAnalysisGroupHandler();

   virtual PV::ParamGroupType getGroupType(char const * keyword);

   virtual PV::HyPerLayer * createLayer(char const * keyword, char const * name, PV::HyPerCol * hc);

   virtual PV::BaseProbe * createProbe(char const * keyword, char const * name, PV::HyPerCol * hc);

}; // class soundAnalysisGroupHandler

#endif // SOUNDANALYSISGROUPHANDLER_HPP_
