/**
 * MLearningGroupHandler.hpp
 *
 * Created on: Mar 18, 2015
 *     Author: pschultz
 */

#ifndef MLEARNINGGROUPHANDLER_HPP_
#define MLEARNINGGROUPHANDLER_HPP_

#include <io/ParamGroupHandler.hpp>

namespace PVMLearning{

class MLearningGroupHandler : public PV::ParamGroupHandler {
public:
   MLearningGroupHandler() {}
   virtual ~MLearningGroupHandler() {}
   
   virtual PV::ParamGroupType getGroupType(char const * keyword);
   
   virtual PV::HyPerLayer * createLayer(char const * keyword, char const * name, PV::HyPerCol * hc);
   virtual PV::BaseConnection * createConnection(char const * keyword, char const * name, PV::HyPerCol * hc, PV::InitWeights * weightInitializer=NULL, PV::NormalizeBase * weightNormalizer=NULL);

}; // class MLearningGroupHandler

}  // namespace PVMLearning

#endif
