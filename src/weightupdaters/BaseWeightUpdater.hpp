/*
 * BaseWeightUpdater.hpp
 *
 *  Created on: Nov 29, 2017
 *      Author: Pete Schultz
 */

#ifndef BASEWEIGHTUPDATER_HPP_
#define BASEWEIGHTUPDATER_HPP_

#include "columns/BaseObject.hpp"
#include "components/ArborList.hpp"
#include "components/ConnectionData.hpp"

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

   virtual void setObjectType() override;

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

  protected:
   bool mPlasticityFlag = true;

   ConnectionData *mConnectionData = nullptr;
   ArborList *mArborList           = nullptr;
};

} // namespace PV

#endif // BASEWEIGHTUPDATER_HPP_
