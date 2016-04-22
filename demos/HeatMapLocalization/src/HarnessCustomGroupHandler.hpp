/*
 * HarnessCustomGroupHandler.hpp
 *
 *  Created on: Sep 23, 2015
 *      Author: pschultz
 */

#ifndef HARNESSCUSTOMGROUPHANDLER_HPP_
#define HARNESSCUSTOMGROUPHANDLER_HPP_

#include <io/ParamGroupHandler.hpp>

class HarnessCustomGroupHandler: public PV::ParamGroupHandler {
public:
   HarnessCustomGroupHandler();
   PV::ParamGroupType getGroupType(char const * keyword);
   virtual PV::HyPerLayer * createLayer(char const * keyword, char const * name, PV::HyPerCol * hc);
   virtual PV::BaseProbe * createProbe(char const * keyword, char const * name, PV::HyPerCol * hc);
   virtual ~HarnessCustomGroupHandler();
};

#endif /* HARNESSCUSTOMGROUPHANDLER_HPP_ */
