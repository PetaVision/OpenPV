/*
 * FirmThresholdCostFnProbe.hpp
 *
 *  Created on: Aug 14, 2015
 *      Author: pschultz
 */

#ifndef FIRMTHRESHOLDCOSTFNPROBE_HPP_
#define FIRMTHRESHOLDCOSTFNPROBE_HPP_

#include "AbstractNormProbe.hpp"

namespace PV {
class FirmThresholdCostFnProbe : public AbstractNormProbe {
public:
   FirmThresholdCostFnProbe(const char * probeName, HyPerCol * hc);
   virtual ~FirmThresholdCostFnProbe();

protected:
   FirmThresholdCostFnProbe();
   int initFirmThresholdCostFnProbe(const char * probeName, HyPerCol * hc);
   virtual double getValueInternal(double timevalue, int index);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_VThresh(enum ParamsIOFlag ioFlag);
   virtual void ioParam_VWidth(enum ParamsIOFlag ioFlag);

private:
   int initFirmThresholdCostFnProbe_base();

// Member variables
protected:
   pvpotentialdata_t VThresh;
   pvpotentialdata_t VWidth;
}; // end class FirmThresholdCostFnProbe

}  // end namespace PV

#endif /* FIRMTHRESHOLDCOSTFNPROBE_HPP_ */
