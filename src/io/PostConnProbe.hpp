/*
 * PostConnProbe.hpp
 *
 *  Created on: May 12, 2009
 *      Author: rasmussn
 */

#ifndef POSTCONNPROBE_HPP_
#define POSTCONNPROBE_HPP_

#include "ConnectionProbe.hpp"
#include "../layers/Image.hpp"

namespace PV {

class PostConnProbe: public PV::ConnectionProbe {
public:
   PostConnProbe(int kPost, int arbID=1);
   PostConnProbe(int kxPost, int kyPost, int kfPost, int arbID=1);
   PostConnProbe(const char * filename, HyPerCol * hc, int kPost, int arbID=1);
   PostConnProbe(const char * filename, HyPerCol * hc, int kxPost, int kyPost, int kfPost, int arbID=1);
   virtual ~PostConnProbe();

   virtual int outputState(float time, HyPerConn * c);
   void setImage(Image * image)   {this->image = image;}

   int text_write_patch_extra(FILE * fp, PVPatch * patch,
                              pvdata_t * data, pvdata_t * prev, pvdata_t * activ);

protected:
   int kPost;   // index of post-synaptic neuron
   int kxPost, kyPost, kfPost;
   Image * image;
   pvdata_t * wPrev;
   pvdata_t * wActiv;
};

}

#endif /* POSTCONNPROBE_HPP_ */
