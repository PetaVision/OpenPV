/*
 * L0NormProbe.hpp
 *
 *  Created on: Aug 11, 2015
 *      Author: pschultz
 */

#ifndef L0NORMPROBE_HPP_
#define L0NORMPROBE_HPP_

#include "AbstractNormProbe.hpp"

namespace PV {

/**
 * A layer probe for returning the number of elements in its target layer's activity buffer
 * above a certain threshold (often referred to as the L0-norm).
 */
class L0NormProbe : public AbstractNormProbe {
public:
   L0NormProbe(const char * probeName, HyPerCol * hc);
   virtual ~L0NormProbe();

protected:
   L0NormProbe();
   int initL0NormProbe(const char * probeName, HyPerCol * hc);
   virtual double getValueInternal(double timevalue, int index);
   
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   /** 
    * List of parameters for the L0NormProbe class
    * @name L0NormProbe Parameters
    * @{
    */

   /**
    * @brief nnzThreshold: The threshold for computing the L0-norm.
    * getValue(t, index) returns the number of targetLayer neurons whose
    * absolute value is greater than nnzThreshold.
    */
   virtual void ioParam_nnzThreshold(enum ParamsIOFlag ioFlag);   
   /** @} */

   /**
    * Overrides AbstractNormProbe::setNormDescription() to set normDescription to "L0-norm".
    * Return values and errno are set by a call to setNormDescriptionToString.
    */
   virtual int setNormDescription();

private:
   int initL0NormProbe_base() {return PV_SUCCESS;}

protected:
   pvadata_t nnzThreshold;
}; // end class L0NormProbe

BaseObject * createL0NormProbe(char const * name, HyPerCol * hc);

}  // end namespace PV

#endif /* L0NORMPROBE_HPP_ */
