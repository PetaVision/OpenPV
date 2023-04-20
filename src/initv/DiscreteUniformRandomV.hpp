/*
 * DiscreteUniformRandomV.hpp
 *
 *  Created on: Sept 28, 2022
 *      Author: peteschultz
 */

#ifndef DISCRETEUNIFORMRANDOMV_HPP_
#define DISCRETEUNIFORMRANDOMV_HPP_

#include "BaseInitV.hpp"

namespace PV {

/**
 * An InitV class that takes the values randomly from a set of evenly spaced values:
 *     minV, minV+dV, minV+2dV, ..., minV + (numValues-1)dV, minV + (numValues-1)*dV, where
 *     dV = (maxV-minV) / (numValues-1).
 */
class DiscreteUniformRandomV : public BaseInitV {
  protected:
   /**
    * List of parameters needed from the DiscreteUniformRandomV class
    * @name DiscreteUniformRandomV Parameters
    * @{
    */

   /**
    * @brief minV: The minimum value of the random distribution. Default value 0.
    */
   virtual void ioParam_minV(enum ParamsIOFlag ioFlag);

   /**
    * @brief maxV: The maximum value of the random distribution. Default value is minV + 1.
    *
    * It is a fatal error for maxV to be less than minV.
    */
   virtual void ioParam_maxV(enum ParamsIOFlag ioFlag);

   /**
    * @breif numValues: The number of possible values the weight can assume.
    *
    * Example: If minV = 1.00 and maxV = 2.00 and numValues = 5,
    * the possible values are 1.00, 1.25, 1.50, 1.75, or 2.00.
    *
    * numValues must be an integer greater than 1.
    */
   virtual void ioParam_numValues(enum ParamsIOFlag ioFlag);
   /** @} */
  public:
   DiscreteUniformRandomV(char const *name, PVParams *params, Communicator const *comm);
   virtual ~DiscreteUniformRandomV();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void calcV(float *V, PVLayerLoc const *loc) override;

  protected:
   DiscreteUniformRandomV();
   void initialize(char const *name, PVParams *params, Communicator const *comm);

  private:
   int initialize_base();

   // data members
  private:
   float mMinV = 0.0f;
   float mMaxV = 1.0f;
   int mNumValues = 2;

}; // end class DiscreteUniformRandomV

} // end namespace PV

#endif /* DISCRETEUNIFORMRANDOMV_HPP_ */
