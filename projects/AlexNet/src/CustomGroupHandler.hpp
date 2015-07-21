/*
 * CustomGroupHandler.hpp
 *
 *  Created on: Mar 2, 2015
 *      Author: pschultz
 */

#ifndef CUSTOMGROUPHANDLER_HPP_
#define CUSTOMGROUPHANDLER_HPP_

#include <io/ParamGroupHandler.hpp>

namespace PV {

class CustomGroupHandler: public ParamGroupHandler {
public:
   CustomGroupHandler();
   virtual ~CustomGroupHandler();
   virtual ParamGroupType getGroupType(char const * keyword);
   virtual HyPerLayer * createLayer(char const * keyword, char const * name, HyPerCol * hc);
   virtual BaseConnection * createConnection(char const * keyword, char const * name, PV::HyPerCol * hc, PV::InitWeights * weightInitializer=NULL, PV::NormalizeBase * weightNormalizer=NULL);
};

} /* namespace PV */

#endif /* CUSTOMGROUPHANDLER_HPP_ */
