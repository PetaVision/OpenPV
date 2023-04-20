/*
 * InitDiscreteUniformRandomWeights.hpp
 *
 *  Created on: Sep 27, 2022
 *      Author: peteschultz
 */

#ifndef INITDISCRETERANDOMWEIGHTS_HPP_
#define INITDISCRETERANDOMWEIGHTS_HPP_

#include "InitRandomWeights.hpp"

namespace PV {

/**
 * An InitWeights class that takes the values randomly from a set of evenly spaced values:
 *     wMin, wMin+dW, wMin+2dW, ..., wMin + (numValues-1)dW, wMin + (numValues-1)*dW, where
 *     dW = (wMax-wMin) / (numValues-1).
 * The parameters wMin, wMax, and numValues are required.
 */
class InitDiscreteUniformRandomWeights : public InitRandomWeights {
  protected:
   /**
    * wMin: The minimum value of the possible weight values
    */
   void ioParam_wMin(enum ParamsIOFlag ioFlag);
   /**
    * wMax: The maximum value of the possible weight values
    */
   void ioParam_wMax(enum ParamsIOFlag ioFlag);
   /**
    * numValues: The number of possible values the weight can assume.
    *
    * Example: If wMin = 1.00 and wMax = 2.00 and numValues = 5,
    * the possible values are 1.00, 1.25, 1.50, 1.75, or 2.00.
    *
    * numValues must be an integer greater than 1.
    */
   void ioParam_wNumValues(enum ParamsIOFlag ioFlag);

  public:
   InitDiscreteUniformRandomWeights(char const *name, PVParams *params, Communicator const *comm);
   virtual ~InitDiscreteUniformRandomWeights();

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

  protected:
   InitDiscreteUniformRandomWeights();
   void initialize(char const *name, PVParams *params, Communicator const *comm);
   virtual void randomWeights(float *patchDataStart, int patchIndex) override;

   // Data members
  protected:
   float mWMin;
   float mWMax;
   int mNumValues;

}; // class InitDiscreteUniformRandomWeights

} /* namespace PV */
#endif /* INITDISCRETERANDOMWEIGHTS_HPP_ */
