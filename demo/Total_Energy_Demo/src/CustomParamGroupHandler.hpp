/*
 * CustomParamGroupHandler.hpp
 *
 *  Created on: Aug 11, 2015
 *      Author: pschultz
 */

#ifndef CUSTOMPARAMGROUPHANDLER_HPP_
#define CUSTOMPARAMGROUPHANDLER_HPP_

#include <io/ParamGroupHandler.hpp>

namespace PV {

class CustomParamGroupHandler: public ParamGroupHandler {
public:
   CustomParamGroupHandler();
   virtual ParamGroupType getGroupType(char const * keyword);
   virtual ~CustomParamGroupHandler();
   virtual BaseProbe * createProbe(char const * keyword, char const * name, HyPerCol * hc);
}; // class CustomParamGroupHandler

} /* namespace PV */

#endif /* CUSTOMPARAMGROUPHANDLER_HPP_ */
