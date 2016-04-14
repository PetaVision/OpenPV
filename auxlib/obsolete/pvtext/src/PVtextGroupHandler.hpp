/**
 * PVtextGroupHandler.hpp
 *
 * Created on: Mar 5, 2015
 *     Author: pschultz
 */

#ifndef PVTEXTGROUPHANDLER_HPP_
#define PVTEXTGROUPHANDLER_HPP_

#include <io/ParamGroupHandler.hpp>

namespace PVtext {

class PVtextGroupHandler : public PV::ParamGroupHandler {
public:
   PVtextGroupHandler() {}
   virtual ~PVtextGroupHandler() {}
   
   virtual PV::ParamGroupType getGroupType(char const * keyword);
   
   virtual PV::HyPerLayer * createLayer(char const * keyword, char const * name, PV::HyPerCol * hc);

   virtual PV::BaseProbe * createProbe(char const * keyword, char const * name, PV::HyPerCol * hc);

}; // class PVtextGroupHandler

}  // namespace PVtext

#endif // PVTEXTGROUPHANDLER_HPP_
