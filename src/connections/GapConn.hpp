/*
 * GapConn.hpp
 *
 *  Created on: Aug 2, 2011
 *      Author: garkenyon
 */

#ifndef GAPCONN_HPP_
#define GAPCONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class GapConn : public PV::HyPerConn {
  public:
   GapConn(const char *name, HyPerCol *hc);
   virtual ~GapConn();
   virtual int allocateDataStructures() override;

  protected:
   GapConn();
   void ioParam_channelCode(enum ParamsIOFlag ioFlag) override;
   // No channel argument in params because GapConn must always use CHANNEL_GAP

   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag) override;
   void ioParam_normalizeMethod(enum ParamsIOFlag ioFlag) override;

   int initialize(const char *name, HyPerCol *hc);

  private:
   int initialize_base();
   bool initNormalizeFlag;

}; // end class GapConn

} /* namespace PV */
#endif /* GAPCONN_HPP_ */
