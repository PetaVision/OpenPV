/*
 * AbstractNormProbe.hpp
 *
 *  Created on: Aug 11, 2015
 *      Author: pschultz
 */

#ifndef ABSTRACTNORMPROBE_HPP_
#define ABSTRACTNORMPROBE_HPP_

#include "LayerProbe.hpp"

namespace PV {

/**
 * An abstract layer probe where getValue and getValues return a norm-like quantity
 * where the quantity over the entire column is the sum of subquantities computed
 * over each MPI process.
 *
 * Derived classes must implement getValueInternal(double, int).  Each MPI process
 * should return its own contribution to the norm.  getValues() and getValue()
 * call getValueInternal and apply MPI_Allreduce to the result, so that
 * getValueInternal() typically does not have to call any MPI processes. 
 */
class AbstractNormProbe : public LayerProbe {
public:
   AbstractNormProbe(const char * probeName, HyPerCol * hc);
   virtual ~AbstractNormProbe();
   
   /**
    * Returns a vector whose length is the HyPerCol's batch size.
    * The kth element of the vector is the norm of the kth batch element of the targetLayer.
    * Derived classes must define the norm in the getValuesInternal() method.
    */
   virtual int getValues(double timevalue, std::vector<double> * values);
   
   /**
    * Returns the norm of the kth batch element of the targetLayer.
    * Derived classes must define the norm in the getValuesInternal() method.
    */
   virtual double getValue(double timevalue, int index);
   
   /**
    * Returns a pointer to the masking layer.  Returns NULL if masking is not used.
    */
   HyPerLayer * getMaskLayer() { return maskLayer; }

   /**
    * Returns the name of the masking layer, as given by the params.
    * maskLayerName is not set in params, it returns NULL.
    */
   char const * getMaskLayerName() { return maskLayerName; }

protected:
   AbstractNormProbe();
   int initAbstractNormProbe(const char * probeName, HyPerCol * hc);
   
   /**
    * Called during initialization, sets the member variable normDescription.
    * This member variable is used by outputState() when printing the norm value.
    * AbstractNormProbe::setNormDescription calls setNormDescriptionToString("norm")
    * and can be overridden.  setNormDescription() returns PV_SUCCESS or PV_FAILURE
    * and on failure it sets errno.
    */
   virtual int setNormDescription();
   
   /**
    * A generic method for setNormDescription() implementations to call.  It
    * frees normDescription if it has already been set, and then copies the
    * string in the input argument to the normDescription member variable.
    * It returns PV_SUCCESS or PV_FAILURE; on failure it sets errno.
    */
   int setNormDescriptionToString(char const * s);
   
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   
   /**
    * List of parameters needed from the AbstractNormProbe class
    * @name AbstractNormProbe Parameters
    * @{
    */

   /**
    * @brief maskLayerName: Specifies a masking layer to use when calculating the norm.
    * When blank (the default), masking is not used.
    * @details The motivation for maskLayerName is to use a layer of
    * ones and zeros to mask out do-not-care regions when computing the norm.
    * Note that for reasons of computation speed, it is up to derived classes to
    * take masking into account when implementing getValueInternal().
    * The maskLayerName must refer to a layer in the HyPerCol with the same nxScale and
    * nyScale as the probe's targetLayer, and have either the same number of features
    * or a single feature.
    */
   virtual void ioParam_maskLayerName(enum ParamsIOFlag ioFlag);
   /** @} */
   
   /**
    * getValueInternal(double, index) is a pure virtual function
    * called by getValue() and getValues().  The index refers to the layer's batch element index.
    *
    * getValue(t, index) returns the sum of each MPI process's getValueInternal(t, index).
    * getValues(t, values) returns a vector whose kth element is the value that getValue(t, k) returns.
    */
   virtual double getValueInternal(double timevalue, int index) = 0;
   
   /**
    * Calls LayerProbe::communicateInitInfo to set up the targetLayer and
    * attach the probe; and then checks the masking layer if masking is used.
    */
   virtual int communicateInitInfo();
   
   /**
    * Returns true if masking is used and the layer has multiple features but
    * the masking layer has a single feature.  In this case it is expected that
    * the single masking feature will be applied to all layer features.
    * Not reliable until communicateInitInfo() has been called.
    */
   bool maskHasSingleFeature() { return singleFeatureMask; }

   /**
    * Implements the outputState method required by classes derived from BaseProbe.
    * Prints to the outputFile the probe message, timestamp, number of neurons, and norm value for each batch element.
    */
   virtual int outputState(double timevalue);
   
   char const * getNormDescription() { return normDescription; }

private:
   int initAbstractNormProbe_base();

private:
   char * normDescription;
   char * maskLayerName;
   HyPerLayer * maskLayer;
   bool singleFeatureMask;
   
   double timeLastComputed; // the value of the input argument timevalue for the most recent getValues() call.  Calls to getValue() do not set or refer to this time.
   std::vector<double> norms; // the values of the norms, as given by the most recent getValues() call.
   
}; // end class AbstractNormProbe

}  // end namespace PV

#endif /* ABSTRACTNORMPROBE_HPP_ */
