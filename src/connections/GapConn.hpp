/*
 * GapConn.hpp
 *
 *  Created on: Aug 2, 2011
 *      Author: garkenyon
 */

#ifndef GAPCONN_HPP_
#define GAPCONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class GapConn : public HyPerConn {
  public:
   GapConn(const char *name, PVParams *params, Communicator const *comm);
   virtual ~GapConn();
}; // end class GapConn

} /* namespace PV */
#endif /* GAPCONN_HPP_ */
