/*
 * UniformRandomV.hpp
 *
 *  Created on: Oct 26, 2016
 *      Author: pschultz
 */

#ifndef UNIFORMRANDOMV_HPP_
#define UNIFORMRANDOMV_HPP_

#include "BaseInitV.hpp"

namespace PV {

class UniformRandomV : public BaseInitV {
  protected:
   /**
    * List of parameters needed from the UniformRandomV class
    * @name UniformRandomV Parameters
    * @{
    */

   /**
    * @brief minV: The minimum value of the random distribution
    */
   virtual void ioParam_minV(enum ParamsIOFlag ioFlag);

   /**
    * @brief maxV: The maximum value of the random distribution
    */
   virtual void ioParam_maxV(enum ParamsIOFlag ioFlag);
   /** @} */
  public:
   UniformRandomV(char const *name, HyPerCol *hc);
   virtual ~UniformRandomV();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void calcV(float *V, PVLayerLoc const *loc) override;

  protected:
   UniformRandomV();
   int initialize(char const *name, HyPerCol *hc);

  private:
   int initialize_base();

   // data members
  private:
   float minV = (float)0;
   float maxV = (float)1;

}; // end class UniformRandomV

} // end namespace PV

#endif /* UNIFORMRANDOMV_HPP_ */
