/*
 * CustomGroupHandler.hpp
 *
 *  Created on: Feb 18, 2015
 *      Author: pschultz
 */

#ifndef CUSTOMGROUPHANDLER_HPP_
#define CUSTOMGROUPHANDLER_HPP_

#include <io/ParamGroupHandler.hpp>

namespace PV {

class CustomGroupHandler : public ParamGroupHandler {
public:
   CustomGroupHandler();

   virtual ~CustomGroupHandler();

   virtual ParamGroupType getGroupType(char const * keyword);

   virtual BaseProbe * createProbe(char const * keyword, char const * name, HyPerCol * hc);
}; /* class CustomGroupHandler */

}  // namespace PV

#endif // CUSTOMGROUPHANDLER_HPP_
