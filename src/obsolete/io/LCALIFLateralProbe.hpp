/*
 * LCALIFLateralProbe.hpp
 *
 *  Created on: Oct 30, 2012
 *      Author: slundquist
 */

#ifndef LCALIFLATERALPROBE_HPP_
#define LCALIFLATERALPROBE_HPP_

#include "BaseConnectionProbe.hpp"
#include "../connections/LCALIFLateralConn.hpp"
#include <assert.h>


namespace PV {
class LCALIFLateralProbe: public BaseConnectionProbe {
   //Methods
public:
   LCALIFLateralProbe();
   LCALIFLateralProbe(const char * probename, HyPerCol * hc);
   // LCALIFLateralProbe(const char * probename, const char * filename, HyPerConn * conn, int kxPre, int kyPre, int kfPre);
   virtual ~LCALIFLateralProbe();
   virtual int allocateDataStructures();

   virtual int outputState(double timef);

protected:
   int initialize(const char * probename, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_kPost(enum ParamsIOFlag ioFlag);
   virtual void ioParam_kxPost(enum ParamsIOFlag ioFlag);
   virtual void ioParam_kyPost(enum ParamsIOFlag ioFlag);
   virtual void ioParam_kfPost(enum ParamsIOFlag ioFlag);

private:
   int initialize_base();
   LCALIFLateralConn * LCALIFConn;
   int kLocalRes;
   int kLocalExt;
   int inBounds;
   pvwdata_t* postWeights;

   //output variables
   float postIntTr;
   pvwdata_t * preWeights;

   PatchIDMethod patchIDMethod;
   int kPost;  // Index of patch
   int kxPost; // x-coordinate of patch
   int kyPost; // y-coordinate of patch
   int kfPost; // feature number of patch

};
} // end namespace PV



#endif /* LCALIFLATERALPROBE_HPP_ */
