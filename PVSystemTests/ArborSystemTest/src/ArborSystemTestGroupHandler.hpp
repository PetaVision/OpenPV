/*
 * ArborSystemTestGroupHandler.hpp
 *
 *  Created on: Dec 30, 2014
 *      Author: pschultz
 */

#ifndef ARBORSYSTEMTESTGROUPHANDLER_HPP_
#define ARBORSYSTEMTESTGROUPHANDLER_HPP_

#include <io/ParamGroupHandler.hpp>

namespace PV {

class ArborSystemTestGroupHandler: public ParamGroupHandler {
public:
   ArborSystemTestGroupHandler();
   virtual ~ArborSystemTestGroupHandler();
   virtual ParamGroupType getGroupType(char const * keyword);
   virtual BaseProbe * createProbe(char const * keyword, char const * name, HyPerCol * hc);
};

} /* namespace PV */

#endif /* ARBORSYSTEMTESTGROUPHANDLER_HPP_ */
