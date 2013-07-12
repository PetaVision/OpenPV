/*
 * PatchProbe.hpp (formerly ConnectionProbe.hpp)
 *
 *  Created on: Apr 25, 2009
 *      Author: rasmussn
 */

#ifndef PATCHPROBE_HPP_
#define PATCHPROBE_HPP_

#include "BaseConnectionProbe.hpp"
#include "../connections/HyPerConn.hpp"

namespace PV {

enum PatchIDMethod { INDEX_METHOD, COORDINATE_METHOD };

class PatchProbe : public BaseConnectionProbe {

// Methods
public:
   PatchProbe(const char * probename, HyPerCol * hc);
   virtual ~PatchProbe();

   virtual int outputState(double timef);

   static int text_write_patch(FILE * fd, int nx, int ny, int nf, int sx, int sy, int sf, float * data);
   static int write_patch_indices(FILE * fp, PVPatch * patch,
                                  const PVLayerLoc * loc, int kx0, int ky0, int kf0);

   int getKPre()                        {return kPre;}
   int getKxPre()                       {return kxPre;}
   int getKyPre()                       {return kyPre;}
   int getKfPre()                       {return kfPre;}
   int getArborID()                     {return arborID;}
   bool getOutputWeightsFlag()          {return outputWeights;}
   bool getOutputPlasticIncrFlag()      {return outputPlasticIncr;}
   bool getOutputPostIndicesFlag()      {return outputPostIndices;}

protected:
   PatchProbe();
   int initialize(const char * probename, HyPerCol * hc);
   virtual int getPatchID();

private:
   int initialize_base();

// Member variables
protected:
   PatchIDMethod patchIDMethod;

private:
   int    kPre;  // index of pre-synaptic neuron
   int    kxPre, kyPre, kfPre;
   int    arborID;
   bool   outputWeights;
   bool   outputPlasticIncr;
   bool   outputPostIndices;
};

} // namespace PV

#endif /* PATCHPROBE_HPP_ */
