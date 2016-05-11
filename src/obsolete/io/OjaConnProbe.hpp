/*
 * OjaConnProbe.hpp
 *
 *  Created on: Oct 15, 2012
 *      Author: dpaiton
 */

#ifndef OJACONNPROBE_HPP_
#define OJACONNPROBE_HPP_

#include "BaseConnectionProbe.hpp"
#include "../connections/OjaSTDPConn.hpp"
#include <assert.h>

//#define DEBUG_POST
#undef DEBUG_POST

namespace PV {

class OjaConnProbe: public BaseConnectionProbe {
   //Methods
public:
   OjaConnProbe();
   OjaConnProbe(const char * probename, HyPerCol * hc);
   virtual ~OjaConnProbe();

   virtual int allocateDataStructures();

   virtual int outputState(double timef);

   static int text_write_patch(FILE * fd, int nx, int ny, int nf, int sx, int sy, int sf, float * data);
   static int write_patch_indices(FILE * fp, PVPatch * patch,
                                  const PVLayerLoc * loc, int kx0, int ky0, int kf0);

protected:
   int initialize(const char * probename, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_kPost(enum ParamsIOFlag ioFlag);
   virtual void ioParam_kxPost(enum ParamsIOFlag ioFlag);
   virtual void ioParam_kyPost(enum ParamsIOFlag ioFlag);
   virtual void ioParam_kfPost(enum ParamsIOFlag ioFlag);

private:
   int initialize_base();
   OjaSTDPConn * ojaConn;
   int kLocal;
   int inBounds;

   //output variables
   float postStdpTr;
   float postOjaTr;
   float postIntTr;
   float ampLTD;
   float * preStdpTrs;
   float * preOjaTrs;
   pvwdata_t * preWeights;
   pvwdata_t ** postWeightsp;
#ifdef DEBUG_POST
   pvwdata_t * preWeightsDebug;
   pvwdata_t * postWeights;
#endif

   PatchIDMethod patchIDMethod;
   int kPost;  // Index of patch
   int kxPost; // x-coordinate of patch
   int kyPost; // y-coordinate of patch
   int kfPost; // feature number of patch
};
} // end namespace PV
#endif /* OJACONNPROBE_HPP_ */
