/*
 * InterColComm.cpp
 *
 *  Created on: Aug 28, 2008
 *      Author: rasmussn
 */

#include "InterColComm.hpp"
#include "utils/PVLog.hpp"

namespace PV {

// InterColComm is obsolete as of Jul 26, 2016.
InterColComm::InterColComm(PV_Arguments * argumentList) {
   pvError() << "InterColComm is obsolete.  Use Communicator class instead.\n";
}

} // end namespace PV
