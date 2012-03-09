/*
 * PoolingANNLayer.hpp
 *
 * The output V is determined from GSynExc and GSynInh
 * using the formula GSynExc*GSynInh*(biasExc*GSynExc+biasInh*GSynInh)
 * biasExc and biasInh are set by the params file parameter bias:
 * biasExc = (1+bias)/2;  biasInh = (1-bias)/2
 *
 * This type of expression arises in the pooling generative models
 * "Exc" and "Inh" are really misnomers for this class, but the
 * terminology is inherited from the base class.
 *
 *  Created on: Apr 20, 2011
 *      Author: peteschultz
 */

#ifndef POOLINGANNLAYER_HPP_
#define POOLINGANNLAYER_HPP_

#include "ANNLayer.hpp"

namespace PV {

class PoolingANNLayer : public ANNLayer {
public:
   PoolingANNLayer(const char * name, HyPerCol * hc);
   int initialize();

   virtual int updateState(float timef, float dt);

   pvdata_t getBiasa() { return biasa;}
   pvdata_t getBiasb() { return biasb;}
   void setBias(pvdata_t bias) { biasa=0.5*(1+bias); biasb=0.5*(1-bias); return;}

protected:
   PoolingANNLayer();
   int initialize(const char * name, HyPerCol * hc);
   /* static */ int updateState(float timef, float dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * gSynHead, pvdata_t biasa, pvdata_t biasb);
   // int updateV();

private:
   int initialize_base();
   pvdata_t biasa;
   pvdata_t biasb;
};

}  // end namespace PV block

#endif /* POOLINGANNLAYER_HPP_ */
