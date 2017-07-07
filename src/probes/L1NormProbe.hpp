/*
 * L1NormProbe.hpp
 *
 *  Created on: Aug 11, 2015
 *      Author: pschultz
 */

#ifndef L1NORMPROBE_HPP_
#define L1NORMPROBE_HPP_

#include "AbstractNormProbe.hpp"

namespace PV {
/**
 * A layer probe for returning the L1-norm of its target layer's activity buffer
 */
class L1NormProbe : public AbstractNormProbe {
  public:
   L1NormProbe(const char *name, HyPerCol *hc);
   virtual ~L1NormProbe();

  protected:
   L1NormProbe();
   int initialize(const char *name, HyPerCol *hc);

   /**
    * For each MPI process, getValueInternal returns the sum of the absolute
    * values of the activities in the restricted space of that MPI process.
    */
   virtual double getValueInternal(double timevalue, int index) override;

   /**
    * Overrides AbstractNormProbe::setNormDescription() to set normDescription to
    * "L1-norm".
    * Return values and errno are set by a call to setNormDescriptionToString.
    */
   virtual int setNormDescription() override;

  private:
   int initialize_base() { return PV_SUCCESS; }
}; // end class L1NormProbe

} // end namespace PV

#endif /* L1NORMPROBE_HPP_ */
