/*
 * CopyUpdater.hpp
 *
 *  Created on: Dec 15, 2017
 *      Author: Pete Schultz
 */

#ifndef COPYUPDATER_HPP_
#define COPYUPDATER_HPP_

#include "components/CopyWeightsPair.hpp"
#include "components/Weights.hpp"
#include "weightupdaters/BaseWeightUpdater.hpp"

namespace PV {

class CopyUpdater : public BaseWeightUpdater {
  protected:
   /**
    * List of parameters needed from the CopyUpdater class
    * @name CopyUpdater Parameters
    * @{
    */

   /**
    * CopyUpdater does not read plasticity from params, but copies it from the
    * original connection's updater
    */
   virtual void ioParam_plasticityFlag(enum ParamsIOFlag ioFlag) override;

   /** @} */ // end of CopyUpdater parameters

  public:
   CopyUpdater(char const *name, PVParams *params, Communicator *comm);

   virtual ~CopyUpdater() {}

  protected:
   CopyUpdater() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType() override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   virtual void updateState(double timestamp, double dt) override;

  protected:
   CopyWeightsPair *mCopyWeightsPair = nullptr;
   Weights *mOriginalWeights         = nullptr;

   bool mWriteCompressedCheckpoints = false;
   double mLastUpdateTime           = 0.0;
};

} // namespace PV

#endif // COPYUPDATER_HPP_
