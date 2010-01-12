/*
 * ConnectionProbe.hpp
 *
 *  Created on: Apr 25, 2009
 *      Author: rasmussn
 */

#ifndef CONNECTIONPROBE_HPP_
#define CONNECTIONPROBE_HPP_

#include "../connections/HyPerConn.hpp"

namespace PV {

class ConnectionProbe {
public:
   ConnectionProbe(int kPre);
   ConnectionProbe(int kxPre, int kyPre, int kfPre);
   ConnectionProbe(const char * filename, int kPre);
   ConnectionProbe(const char * filename,int kxPre, int kyPre, int kfPre);
   virtual ~ConnectionProbe();

   virtual int outputState(float time, HyPerConn * c);

   static int text_write_patch(FILE * fd, PVPatch * patch, float * data);

   static int write_patch_indices(FILE * fp, PVPatch * patch,
                                  const PVLayerLoc * loc, int kx0, int ky0, int kf0);

   void setOutputIndices(bool flag)   {outputIndices = flag;}

protected:
   FILE * fp;
   int    kPre;  // index of pre-synaptic neuron
   int    kxPre, kyPre, kfPre;
   bool   outputIndices;
};

} // namespace PV

#endif /* CONNECTIONPROBE_HPP_ */
