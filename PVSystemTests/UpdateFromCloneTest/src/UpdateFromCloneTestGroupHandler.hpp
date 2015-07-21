/*
 * UpdateFromCloneTestHandler.hpp
 *
 *  Created on: Mar 2, 2015
 *      Author: pschultz
 */

#ifndef UPDATEFROMCLONETESTGROUPHANDLER_HPP_
#define UPDATEFROMCLONETESTGROUPHANDLER_HPP_

#include <io/ParamGroupHandler.hpp>

namespace PV {

class UpdateFromCloneTestGroupHandler: public ParamGroupHandler {
public:
   UpdateFromCloneTestGroupHandler();
   virtual ~UpdateFromCloneTestGroupHandler();
   virtual ParamGroupType getGroupType(char const * keyword);
   virtual BaseProbe * createProbe(char const * keyword, char const * name, HyPerCol * hc);
};

} /* namespace PV */

#endif /* KERNELTESTGROUPHANDLER_HPP_ */
