/*
 * ShuffleLayer.hpp
 * can be used to implement Sigmoid junctions
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#ifndef SHUFFLELAYER_HPP_
#define SHUFFLELAYER_HPP_

#include "HyPerLayer.hpp"

namespace PV {

// CloneLayer can be used to implement Sigmoid junctions between spiking neurons
class ShuffleLayer: public HyPerLayer {
public:
   ShuffleLayer(const char * name, HyPerCol * hc);
   virtual ~ShuffleLayer();
   virtual int allocateDataStructures();
   virtual int updateState(double timef, double dt);
   HyPerLayer * sourceLayer;
   virtual int setActivity();
protected:
   ShuffleLayer();
   int initialize(const char * name, HyPerCol * hc);
   int setParams(PVParams * params);
   void readOriginalLayerName(PVParams * params);
   void readShuffleMethod(PVParams * params);
   void randomShuffle();

private:
   int initialize_base();
   char * originalLayerName;
   HyPerLayer * originalLayer;
   char * shuffleMethod;
   //Mapping of restricted global indicies
   int * indexArray;

};

}

#endif /* CLONELAYER_HPP_ */
