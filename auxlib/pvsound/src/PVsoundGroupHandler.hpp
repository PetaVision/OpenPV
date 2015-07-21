/**
 * PVsoundGroupHandler.hpp
 *
 * Created on: Mar 5, 2015
 *     Author: pschultz
 */

#ifndef PVSOUNDGROUPHANDLER_HPP_
#define PVSOUNDGROUPHANDLER_HPP_

#include <io/ParamGroupHandler.hpp>

namespace PVsound {

class PVsoundGroupHandler : public PV::ParamGroupHandler {
public:
   PVsoundGroupHandler() {}
   virtual ~PVsoundGroupHandler() {}
   
   virtual PV::ParamGroupType getGroupType(char const * keyword);
   
   virtual PV::HyPerLayer * createLayer(char const * keyword, char const * name, PV::HyPerCol * hc);

}; // class PVsoundGroupHandler

}  // namespace PVsound

#endif // PVSOUNDGROUPHANDLER_HPP_
