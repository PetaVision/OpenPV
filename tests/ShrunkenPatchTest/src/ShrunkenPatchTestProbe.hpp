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

class ShrunkenPatchTestProbe : public PV::StatsProbe {
  public:
   ShrunkenPatchTestProbe(const char *probename, PVParams *params, Communicator *comm);
   ShrunkenPatchTestProbe(const char *probename, HyPerLayer *layer, const char *msg);

   virtual Response::Status outputState(double simTime, double deltaTime) override;

   virtual ~ShrunkenPatchTestProbe();

  protected:
   void initialize(const char *probename, PVParams *params, Communicator *comm);
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
