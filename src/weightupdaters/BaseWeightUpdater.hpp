/*
 * BaseWeightUpdater.hpp
 *
 *  Created on: Nov 29, 2017
 *      Author: Pete Schultz
 */

#ifndef BASEWEIGHTUPDATER_HPP_
#define BASEWEIGHTUPDATER_HPP_

#include "columns/BaseObject.hpp"
#include "components/ConnectionData.hpp"
#include "layers/HyPerLayer.hpp"

namespace PV {

class BaseWeightUpdater : public BaseObject {
  protected:
   /**
    * List of parameters needed from the BaseWeightUpdater class
    * @name BaseWeightUpdater Parameters
    * @{
    */

   /**
    * @brief plasticityFlag: Specifies if the weights will update
    */
   virtual void ioParam_plasticityFlag(enum ParamsIOFlag ioFlag);
   /** @} */ // end of BaseWeightUpdater parameters

  public:
   BaseWeightUpdater(char const *name, HyPerCol *hc);

   virtual ~BaseWeightUpdater() {}

   virtual void updateState(double timestamp, double dt) {}

   bool getPlasticityFlag() const { return mPlasticityFlag; };

  protected:
   BaseWeightUpdater() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual int setDescription() override;

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   int communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

  protected:
   bool mPlasticityFlag = true;

   ConnectionData *mConnectionData = nullptr;
};

} // namespace PV

#endif // BASEWEIGHTUPDATER_HPP_