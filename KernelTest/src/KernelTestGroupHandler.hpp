/*
 * KernelTestGroupHandler.hpp
 *
 *  Created on: Dec 30, 2014
 *      Author: pschultz
 */

#ifndef KERNELTESTGROUPHANDLER_HPP_
#define KERNELTESTGROUPHANDLER_HPP_

#include <io/ParamGroupHandler.hpp>

namespace PV {

class KernelTestGroupHandler: public ParamGroupHandler {
public:
   KernelTestGroupHandler();
   virtual ~KernelTestGroupHandler();
   virtual void * createObject(char const * keyword, char const * name, HyPerCol * hc);
};

} /* namespace PV */

#endif /* KERNELTESTGROUPHANDLER_HPP_ */
