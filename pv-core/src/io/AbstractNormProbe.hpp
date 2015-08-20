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
   
   /**
    * getValueInternal(double, index) is a pure virtual function
    * called by getValue() and getValues().  The index refers to the layer's batch element index.
    *
    * getValue(t, index) returns the sum of each MPI process's getValueInternal(t, index).
    * getValues(t, values) returns a vector whose kth element is the value that getValue(t, k) returns.
    */
   virtual double getValueInternal(double timevalue, int index) = 0;
   
   /**
    * Implements the outputState method required by classes derived from BaseProbe.
    * Prints to the outputFile the probe message, timestamp, number of neurons, and norm value for each batch element.
    */
   virtual int outputState(double timevalue);
   
   char const * getNormDescription() { return normDescription; }

private:
   int initAbstractNormProbe_base() {return PV_SUCCESS;}

private:
   char * normDescription;
}; // end class AbstractNormProbe

}  // end namespace PV

#endif /* ABSTRACTNORMPROBE_HPP_ */
