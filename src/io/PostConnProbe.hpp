/*
 * PostConnProbe.hpp
 *
 *  Created on: May 12, 2009
 *      Author: rasmussn
 */

#ifndef POSTCONNPROBE_HPP_
#define POSTCONNPROBE_HPP_

#include "PatchProbe.hpp"       // #include "ConnectionProbe.hpp"
#include "../layers/Image.hpp"

namespace PV {

class PostConnProbe: public PV::PatchProbe {
public:
   PostConnProbe(const char * filename, HyPerConn * conn, int kPost, int arbID=0);
   PostConnProbe(const char * filename, HyPerConn * conn, int kxPost, int kyPost, int kfPost, int arbID=0);
   virtual ~PostConnProbe();

   virtual int outputState(float timef);
   void setImage(Image * image)   {this->image = image;}

   int text_write_patch_extra(FILE * fp, PVPatch * patch,
                              pvdata_t * data, pvdata_t * prev, pvdata_t * activ);

protected:
   int initialize(const char * probename, const char * filename, HyPerConn * conn, PatchIDMethod method, int kPost, int kxPost, int kyPost, int kfPost, int arbID);

protected:
   int kPost;   // index of post-synaptic neuron
   int kxPost, kyPost, kfPost;
   Image * image;
   pvdata_t * wPrev;
   pvdata_t * wActiv;
   bool   outputIndices;
   bool   stdpVars;
};

}

#endif /* POSTCONNPROBE_HPP_ */
