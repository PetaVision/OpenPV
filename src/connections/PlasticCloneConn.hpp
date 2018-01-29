/*
 * PlasticCloneConn.hpp
 *
 *  Created on: May 24, 2011
 *      Author: peteschultz
 */

#ifndef PLASTICCLONECONN_HPP_
#define PLASTICCLONECONN_HPP_

#include "CloneConn.hpp"

namespace PV {

/**
 * A CloneConn that also adds its connection data to the Hebbian updater of the connection
 * specified in OriginalConnName. It is an error for this connection not to have a HebbianUpdater or
 * HebbianUpdater-derived component.
 */
class PlasticCloneConn : public CloneConn {
  protected:
   /**
    * List of parameters needed from the PlasticCloneConn class
    * @name PlasticCloneConn Parameters
    * @{
    */

  public:
   PlasticCloneConn(const char *name, HyPerCol *hc);
   virtual ~PlasticCloneConn();

  protected:
   PlasticCloneConn();
   int initialize(const char *name, HyPerCol *hc);

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
}; // end class PlasticCloneConn

} // end namespace PV

#endif /* CLONECONN_HPP_ */
