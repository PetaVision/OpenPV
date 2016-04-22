/*
 * CustomGroupHandler.hpp
 *
 *  Created on: Sep 23, 2015
 *      Author: pschultz
 */

#ifndef CUSTOMGROUPHANDLER_HPP_
#define CUSTOMGROUPHANDLER_HPP_

#include <io/ParamGroupHandler.hpp>

class PASCALCustomGroupHandler: public PV::ParamGroupHandler {
public:
   PASCALCustomGroupHandler();
   PV::ParamGroupType getGroupType(char const * keyword);
   virtual PV::BaseProbe * createProbe(char const * keyword, char const * name, PV::HyPerCol * hc);
   virtual ~PASCALCustomGroupHandler();
};

#endif /* CUSTOMGROUPHANDLER_HPP_ */
