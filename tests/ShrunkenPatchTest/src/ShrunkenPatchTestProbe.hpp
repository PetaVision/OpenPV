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

   virtual Response::Status outputState(double timestamp) override;

   virtual ~ShrunkenPatchTestProbe();

  protected:
   int initialize(const char *probename, HyPerCol *hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_nxpShrunken(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nypShrunken(enum ParamsIOFlag ioFlag);

  private:
   int initialize_base();

  protected:
   int nxpShrunken;
   int nypShrunken;
   float *correctValues;
}; // end class ShrunkenPatchTestProbe

} // end namespace PV

#endif /* SHRUNKENPATCHTESTPROBE_HPP_ */
