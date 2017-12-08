/*
 * NoCheckpointConnectionData.hpp
 *
 *  Created on: Dec 7, 2017
 *      Author: pschultz
 */

#ifndef NOCHECKPOINTCONNECTIONDATA_HPP_
#define NOCHECKPOINTCONNECTIONDATA_HPP_

#include "components/ConnectionData.hpp"

namespace PV {

/**
 * This component is identical to ConnectionData, except that it does not
 * read initializeFromCheckpointFlag from params, but instead sets it to
 * false. It is an error for a connection that uses a NoCheckpointConnection
 * component to set initializeCheckpointFlag to a non-false value in params.
 */
class NoCheckpointConnectionData : public ConnectionData {
  protected:
   /**
    * @brief initializeFromCheckpointFlag is always set to false.
    */
   virtual void ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag) override;
   /** @} */ // end of BaseDelivery parameters

  public:
   NoCheckpointConnectionData(char const *name, HyPerCol *hc);
   virtual ~NoCheckpointConnectionData();

   virtual int setDescription() override;

  protected:
   NoCheckpointConnectionData();

   int initialize(char const *name, HyPerCol *hc);
}; // class NoCheckpointConnectionData

} // namespace PV

#endif // NOCHECKPOINTCONNECTIONDATA_HPP_
