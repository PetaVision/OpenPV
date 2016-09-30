/*
 * CloneKernelConn.hpp
 *
 *  Created on: May 24, 2011
 *      Author: peteschultz
 */

#ifndef CLONEKERNELCONN_HPP_
#define CLONEKERNELCONN_HPP_

#include "CloneConn.hpp"

namespace PV {

class CloneKernelConn : public CloneConn {

  public:
   CloneKernelConn(const char *name, HyPerCol *hc);
   virtual ~CloneKernelConn();

  protected:
   CloneKernelConn();
   int initialize(const char *name, HyPerCol *hc);

}; // end class CloneKernelConn

} // end namespace PV

#endif /* CLONEKERNELCONN_HPP_ */
