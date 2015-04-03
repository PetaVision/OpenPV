/**
 * MatchingPursuitGroupHandler.hpp
 *
 * Created on: Mar 18, 2015
 *     Author: pschultz
 */

#ifndef MATCHINGPURSUITGROUPHANDLER_HPP_
#define MATCHINGPURSUITGROUPHANDLER_HPP_

#include <io/ParamGroupHandler.hpp>

namespace PVMatchingPursuit {

class MatchingPursuitGroupHandler : public PV::ParamGroupHandler {
public:
   MatchingPursuitGroupHandler() {}
   virtual ~MatchingPursuitGroupHandler() {}
   
   virtual PV::ParamGroupType getGroupType(char const * keyword);
   
   virtual PV::HyPerLayer * createLayer(char const * keyword, char const * name, PV::HyPerCol * hc);

}; // class MatchingPursuitGroupHandler

}  // namespace PVMatchingPursuit

#endif // MATCHINGPURSUITGROUPHANDLER_HPP_
