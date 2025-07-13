/*
 * GaussianRandomV.hpp
 *
 *  Created on: Oct 26, 2016
 *      Author: pschultz
 */

#ifndef GAUSSIANRANDOMV_HPP_
#define GAUSSIANRANDOMV_HPP_

#include "BaseInitV.hpp"

namespace PV {

class GaussianRandomV : public BaseInitV {
  protected:
   /**
    * List of parameters needed from the GaussianRandomV class
    * @name GaussianRandomV Parameters
    * @{
    */

   /**
    * @brief meanV: The mean of the random distribution
    */
   virtual void ioParam_meanV(ParamsIOSwitch ioSwitch);

   /**
    * @brief sigmaV: The standard deviation of the random distribution
    */
   virtual void ioParam_sigmaV(ParamsIOSwitch ioSwitch);
   /** @} */
  public:
   GaussianRandomV(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ~GaussianRandomV();
   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;
   virtual void calcV(float *V, PVLayerLoc const *loc) override;

  protected:
   GaussianRandomV();
   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

  private:
   int initialize_base();
   float meanV  = (float)0;
   float sigmaV = (float)1;

}; // end class GaussianRandomV

} // end namespace PV

#endif /* GAUSSIANRANDOMV_HPP_ */
