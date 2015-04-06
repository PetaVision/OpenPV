/**
 * CustomGroupHandler.hpp
 *
 * Created on: Mar 18, 2015
 *     Author: pschultz
 */

#ifndef CUSTOMGROUPHANDLER_HPP_
#define CUSTOMGROUPHANDLER_HPP_

#include <io/ParamGroupHandler.hpp>

namespace PVMLearning{

class CustomGroupHandler : public PV::ParamGroupHandler {
public:
   CustomGroupHandler() {}
   virtual ~CustomGroupHandler() {}
   
   virtual PV::ParamGroupType getGroupType(char const * keyword);
   
   virtual PV::HyPerLayer * createLayer(char const * keyword, char const * name, PV::HyPerCol * hc);

}; // class CustomGroupHandler

}  // namespace PVMLearning

#endif
