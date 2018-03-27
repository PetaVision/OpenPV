/*
 * InitUniformWeights.hpp
 *
 *  Created on: Aug 23, 2011
 *      Author: kpeterson
 */

#ifndef INITUNIFORMWEIGHTS_HPP_
#define INITUNIFORMWEIGHTS_HPP_

#include "InitWeights.hpp"

namespace PV {

/**
 * A weight initializer that sets each weight to the same value.
 */
class InitUniformWeights : public PV::InitWeights {
  protected:
   /**
    * List of parameters needed by InitUniformWeights class
    * @name InitUniformWeights Parameters
    * @{
    */

   /**
    * @brief weightInit: The value of each weight.
    */
   virtual void ioParam_weightInit(enum ParamsIOFlag ioFlag);

   /**
    * @brief connectOnlySameFeatures: If this flag is set to false,
    * all weights are set to the weightInit value, regardless of pre- and
    * post- feature. If the flag is set to true, the weights are zero unless
    * the pre- and post- feature indices are the same.
    */
   virtual void ioParam_connectOnlySameFeatures(enum ParamsIOFlag ioFlag);
   /** @} */

  public:
   InitUniformWeights(const char *name, HyPerCol *hc);
   virtual ~InitUniformWeights();

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   float getWeightInit() const { return mWeightInit; }
   bool getConnectOnlySameFeatures() const { return mConnectOnlySameFeatures; }

  protected:
   InitUniformWeights();
   int initialize(const char *name, HyPerCol *hc);
   virtual void calcWeights(int patchIndex, int arborId) override;

  private:
   void uniformWeights(float *dataStart, float weightInit, int kf, bool connectOnlySameFeatures);

  private:
   float mWeightInit             = 0.0f;
   bool mConnectOnlySameFeatures = false;

}; // class InitUniformWeights

} /* namespace PV */
#endif /* INITUNIFORMWEIGHTS_HPP_ */
