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
   virtual void ioParam_meanV(enum ParamsIOFlag ioFlag);

   /**
    * @brief sigmaV: The standard deviation of the random distribution
    */
   virtual void ioParam_sigmaV(enum ParamsIOFlag ioFlag);
   /** @} */
  public:
   GaussianRandomV(char const *name, HyPerCol *hc);
   virtual ~GaussianRandomV();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual int calcV(pvdata_t *V, PVLayerLoc const *loc) override;

  protected:
   GaussianRandomV();
   int initialize(char const *name, HyPerCol *hc);

  private:
   int initialize_base();
   pvdata_t meanV = (pvdata_t) 0;
   pvdata_t sigmaV = (pvdata_t) 1;

}; // end class GaussianRandomV

} // end namespace PV

#endif /* GAUSSIANRANDOMV_HPP_ */
