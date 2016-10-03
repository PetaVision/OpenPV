/*
 * customStatsProbe.hpp
 *
 *  Created on: Mar 10, 2009
 *      Author: garkenyon
 */

#ifndef SHRUNKENPATCHTESTPROBE_HPP_
#define SHRUNKENPATCHTESTPROBE_HPP_

#include "probes/StatsProbe.hpp"

namespace PV {

class PVParams;

class ShrunkenPatchTestProbe : public PV::StatsProbe {
  public:
   ShrunkenPatchTestProbe(const char *probename, HyPerCol *hc);
   ShrunkenPatchTestProbe(const char *probename, HyPerLayer *layer, const char *msg);

   virtual int outputState(double timed);

   virtual ~ShrunkenPatchTestProbe();

  protected:
   int initShrunkenPatchTestProbe(const char *probename, HyPerCol *hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nxpShrunken(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nypShrunken(enum ParamsIOFlag ioFlag);

  private:
   int initShrunkenPatchTestProbe_base();

  protected:
   char *probeName;
   int nxpShrunken;
   int nypShrunken;
   pvdata_t *correctValues;
}; // end class ShrunkenPatchTestProbe

} // end namespace PV

#endif /* SHRUNKENPATCHTESTPROBE_HPP_ */
