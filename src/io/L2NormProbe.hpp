/*
 * L2NormProbe.hpp
 *
 *  Created on: Aug 11, 2015
 *      Author: pschultz
 */

#ifndef L2NORMPROBE_HPP_
#define L2NORMPROBE_HPP_

#include "AbstractNormProbe.hpp"

namespace PV {

/**
 * A layer probe for returning the L2-norm of its target layer's activity, raised to
 * a power (set by the exponent parameter).
 */
class L2NormProbe : public AbstractNormProbe {
public:
   L2NormProbe(const char * probeName, HyPerCol * hc);
   virtual ~L2NormProbe();

protected:
   L2NormProbe();
   int initL2NormProbe(const char * probeName, HyPerCol * hc);
   
   /**
    * Overrides AbstractNormProbe::setNormDescription().
    * If exponent == 1.0, normDescription is set to "L2-Norm".
    * If exponent == 2.0, normDescription is set to "L2-Norm squared".
    * Otherwise, it is set to "(L2-Norm)^exp", with "exp" replaced by
    * the value of exponent.
    * Return values and errno are set by a call to setNormDescriptionToString.
    */
   virtual int setNormDescription();
   
   /**
    * Overrides AbstractNormProbe::calcValues method to apply the exponent.
    */
   virtual int calcValues(double timevalue);
   
   /**
    * Each MPI process returns the sum of the squares of the activities in its
    * restricted activity space.  Note that the exponent parameter is not applied
    * inside the call to getValueInternal.
    */
   virtual double getValueInternal(double timevalue, int index);
   
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   /** 
    * List of parameters for the L2NormProbe class
    * @name L2NormProbe Parameters
    * @{
    */

   /**
    * @brief exponent: The exponent on the L2-norm.
    * getValue(t, index) returns (L2-Norm)^exponent.
    * @details (e.g. when exponent=2, getValue returns the sum of the squares;
    * when exponent=1, getValue returns the square root of the sum of the squares.)
    * default is 1.
    */
   virtual void ioParam_exponent(enum ParamsIOFlag ioFlag);
   /** @} */

private:
   int initL2NormProbe_base();

// Member variables
   double exponent;
}; // end class L2NormProbe

BaseObject * createL2NormProbe(char const * name, HyPerCol * hc);

}  // end namespace PV

#endif /* L2NORMPROBE_HPP_ */
