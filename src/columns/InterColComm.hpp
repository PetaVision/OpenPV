/*
 * InterColComm.h
 *
 *  Created on: Aug 28, 2008
 *      Author: rasmussn
 */

#ifndef INTERCOLCOMM_HPP_
#define INTERCOLCOMM_HPP_

#include "columns/PV_Arguments.hpp"
#include "include/pv_common.h"

namespace PV {

// InterColComm is obsolete as of Jul 26, 2016.
class InterColComm {

  public:
   InterColComm(PV_Arguments *argumentList);
};

} // namespace PV

#endif /* INTERCOLCOMM_HPP_ */
