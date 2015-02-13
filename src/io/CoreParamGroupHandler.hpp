/*
 * CoreParamGroupHandler.hpp
 *
 *  Created on: Jan 5, 2015
 *      Author: pschultz
 */

#ifndef COREPARAMGROUPHANDLER_HPP_
#define COREPARAMGROUPHANDLER_HPP_

#include "ParamGroupHandler.hpp"

namespace PV {

class CoreParamGroupHandler: public ParamGroupHandler {
public:
   CoreParamGroupHandler();
   virtual ParamGroupType getGroupType(char const * keyword);
   virtual ~CoreParamGroupHandler();
   virtual HyPerCol * createHyPerCol(char const * keyword, char const * name, HyPerCol * hc);
   virtual HyPerLayer * createLayer(char const * keyword, char const * name, HyPerCol * hc);
   virtual BaseConnection * createConnection(char const * keyword, char const * name, HyPerCol * hc, InitWeights * weightInitializer=NULL, NormalizeBase * weightNormalizer=NULL);
   virtual ColProbe * createColProbe(char const * keyword, char const * name, HyPerCol * hc);
   virtual BaseProbe * createProbe(char const * keyword, char const * name, HyPerCol * hc);
   virtual InitWeights * createWeightInitializer(char const * keyword, char const * name, HyPerCol * hc);
   virtual NormalizeBase * createWeightNormalizer(char const * keyword, char const * name, HyPerCol * hc);
}; // class CoreParamGroupHandler

} /* namespace PV */

#endif /* COREPARAMGROUPHANDLER_HPP_ */
