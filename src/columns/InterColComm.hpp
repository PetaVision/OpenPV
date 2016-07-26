/*
 * InterColComm.h
 *
 *  Created on: Aug 28, 2008
 *      Author: rasmussn
 */

#ifndef INTERCOLCOMM_HPP_
#define INTERCOLCOMM_HPP_

#include "Communicator.hpp"
#include "include/pv_common.h"

namespace PV {

class InterColComm : public Communicator {

public:
   InterColComm(PV_Arguments * argumentList);
   virtual ~InterColComm();
};

} // namespace PV

#endif /* INTERCOLCOMM_HPP_ */
