/*
 * ParamGroupHandler.hpp
 *
 *  Created on: Dec 30, 2014
 *      Author: pschultz
 */

#ifndef PARAMGROUPHANDLER_HPP_
#define PARAMGROUPHANDLER_HPP_

namespace PV {

class HyPerCol;

class ParamGroupHandler {
public:
   ParamGroupHandler();
   virtual ~ParamGroupHandler();
   virtual void * createObject(char const * keyword, char const * name, HyPerCol * hc) = 0;
};

} /* namespace PV */

#endif /* PARAMGROUPHANDLER_HPP_ */
