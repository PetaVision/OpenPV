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

enum PatchIDMethod { INDEX_METHOD, COORDINATE_METHOD };

namespace PV {

class OjaConnProbe: public BaseConnectionProbe {
   //Methods
public:
   OjaConnProbe();
   OjaConnProbe(const char * probename, const char * filename, HyPerConn * conn, int postIndex);
   OjaConnProbe(const char * probename, const char * filename, HyPerConn * conn, int kxPost, int kyPost, int kfPost);
   virtual ~OjaConnProbe();

   virtual int outputState(float timef);

   static int text_write_patch(FILE * fd, int nx, int ny, int nf, int sx, int sy, int sf, float * data);
   static int write_patch_indices(FILE * fp, PVPatch * patch,
                                  const PVLayerLoc * loc, int kx0, int ky0, int kf0);

protected:
   int initialize(const char * probename, const char * filename, HyPerConn * conn, PatchIDMethod method, int postIndex, int kxPost, int kyPost, int kfPost);

private:
   int initialize_base();
   OjaSTDPConn * ojaConn;
   const PVLayerLoc * postLoc;
   int kLocal;
   int inBounds;

   //output variables
   float postStdpTr;
   float postOjaTr;
   float postIntTr;
   float ampLTD;
   float * preStdpTrs;
   float * preOjaTrs;
   float * preWeights;
   pvdata_t ** postWeights;

};
} // end namespace PV
#endif /* OJACONNPROBE_HPP_ */
