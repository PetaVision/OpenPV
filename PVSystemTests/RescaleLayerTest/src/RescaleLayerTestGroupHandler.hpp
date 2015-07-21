/*
 * RescaleLayerTestGroupHandler.hpp
 *
 *  Created on: Dec 30, 2014
 *      Author: pschultz
 */

#ifndef RESCALELAYERTESTGROUPHANDLER_HPP_
#define RESCALELAYERTESTGROUPHANDLER_HPP_

#include <io/ParamGroupHandler.hpp>

namespace PV {

class RescaleLayerTestGroupHandler: public ParamGroupHandler {
public:
   RescaleLayerTestGroupHandler();
   virtual ~RescaleLayerTestGroupHandler();
   virtual ParamGroupType getGroupType(char const * keyword);
   virtual BaseProbe * createProbe(char const * keyword, char const * name, HyPerCol * hc);
};

} /* namespace PV */

#endif /* RESCALELAYERTESTGROUPHANDLER_HPP_ */
