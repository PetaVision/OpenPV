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
   PostConnProbe(const char * filename, HyPerCol * hc);
   virtual ~PostConnProbe();

   virtual int outputState(double timef);
   void setImage(Image * image)   {this->image = image;}

   int text_write_patch_extra(FILE * fp, PVPatch * patch,
                              pvwdata_t * data, pvwdata_t * prev, pvwdata_t * activ, HyPerConn * parentConn);

protected:
   int initialize(const char * probename, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_kPre(enum ParamsIOFlag ioFlag) {return;}
   virtual void ioParam_kxPre(enum ParamsIOFlag ioFlag) {return;}
   virtual void ioParam_kyPre(enum ParamsIOFlag ioFlag) {return;}
   virtual void ioParam_kfPre(enum ParamsIOFlag ioFlag) {return;}
   virtual void ioParam_kPost(enum ParamsIOFlag ioFlag);
   virtual void ioParam_kxPost(enum ParamsIOFlag ioFlag);
   virtual void ioParam_kyPost(enum ParamsIOFlag ioFlag);
   virtual void ioParam_kfPost(enum ParamsIOFlag ioFlag);
   int getPatchID();

protected:
   int kPost;   // index of post-synaptic neuron
   int kxPost, kyPost, kfPost;
   Image * image;
   pvwdata_t * wPrev;
   pvwdata_t * wActiv;
   bool   outputIndices;
   bool   stdpVars;
};

}

#endif /* POSTCONNPROBE_HPP_ */
