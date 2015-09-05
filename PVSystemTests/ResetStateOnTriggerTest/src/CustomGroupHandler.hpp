/*
 * CustomGroupHandler.hpp
 *
 *  Created on: Sep 3, 2015
 *      Author: peteschultz
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
   virtual BaseProbe * createProbe(char const * keyword, char const * name, HyPerCol * hc);
};

} /* namespace PV */

#endif /* CUSTOMGROUPHANDLER_HPP_ */
