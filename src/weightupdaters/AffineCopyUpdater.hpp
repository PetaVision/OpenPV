/*
 * AffineCopyUpdater.hpp
 *
 *  Created on: July 12, 2019
 *      Author: Xinhua Zhang
 */

#ifndef AFFINECOPYUPDATER_HPP_
#define AFFINECOPYUPDATER_HPP_

#include "components/AffineCopyWeightsPair.hpp"
#include "components/Weights.hpp"
#include "weightupdaters/BaseWeightUpdater.hpp"

namespace PV {

class AffineCopyUpdater : public BaseWeightUpdater {
  public:
   AffineCopyUpdater(char const *name, PVParams *params, Communicator const *comm);

   virtual ~AffineCopyUpdater() {}

  protected:
   AffineCopyUpdater() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   virtual void updateState(double timestamp, double dt) override;

  protected:
   AffineCopyWeightsPair *mCopyWeightsPair = nullptr;
   Weights *mOriginalWeights         = nullptr;

   bool mWriteCompressedCheckpoints = false;
   double mLastUpdateTime           = 0.0;
};

} // namespace PV

#endif // AFFINECOPYUPDATER_HPP_
