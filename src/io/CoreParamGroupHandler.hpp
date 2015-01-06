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
   virtual ~CoreParamGroupHandler();
   virtual void * createObject(char const * keyword, char const * name, HyPerCol * hc);
}; // class CoreParamGroupHandler

} /* namespace PV */

#endif /* COREPARAMGROUPHANDLER_HPP_ */
