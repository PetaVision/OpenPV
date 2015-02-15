/*
 * InitWeightsTestParamGroupHandler.hpp
 *
 *  Created on: Feb 13, 2015
 *      Author: pschultz
 */

#ifndef INITWEIGHTSTESTPARAMGROUPHANDLER_HPP_
#define INITWEIGHTSTESTPARAMGROUPHANDLER_HPP_

#include <io/ParamGroupHandler.hpp>

namespace PV {

class InitWeightsTestParamGroupHandler: public ParamGroupHandler {
public:
   InitWeightsTestParamGroupHandler();
   virtual ~InitWeightsTestParamGroupHandler();
   virtual ParamGroupType getGroupType(char const * keyword);
   virtual BaseConnection * createConnection(char const * keyword, char const * name, HyPerCol * hc, InitWeights * weightInitializer=NULL, NormalizeBase * weightNormalizer=NULL);
   virtual BaseProbe * createProbe(char const * keyword, char const * name, HyPerCol * hc);
   virtual InitWeights * createWeightInitializer(char const * keyword, char const * name, HyPerCol * hc);
};

} /* namespace PV */

#endif /* INITWEIGHTSTESTPARAMGROUPHANDLER_HPP_ */
