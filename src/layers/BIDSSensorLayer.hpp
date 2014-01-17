/*
 * BIDSSensorLayer.hpp
 *
 *  Created on: Sep 7, 2012
 *      Author: slundquist
 */

#ifndef BIDSSENSORLAYER_HPP_
#define BIDSSENSORLAYER_HPP_

#include "HyPerLayer.hpp"
#include "BIDSMovieCloneMap.hpp"
#include <math.h>
//For testing
#include <iostream>
#include <fstream>

namespace PV{

class BIDSSensorLayer : public Image{
public:
   BIDSSensorLayer(const char * name, HyPerCol * hc, int numChannels);
   BIDSSensorLayer(const char * name, HyPerCol * hc);
   virtual ~BIDSSensorLayer();
   int initialize_base();
   int initialize(const char * name, HyPerCol * hc, int numChannels=MAX_CHANNELS);
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
   bool needUpdate(double time, double dt);
   int updateState(double timef, double dt);
protected:
   BIDSSensorLayer();
   float matchFilter(int node_index, int frame_index);
   void writeCSV(std::string fname, int node_index);
   BIDSCoords * getCoords() {return blayer->getCoords();}
   int getNumNodes() {return blayer->getNumNodes();}
//   float perfectMatch();
   float** data;
   // BIDSCoords* coords; // Replaced with getNumNodes()
   int buf_size;
   float neutral_val;
   char * blayerName;
   BIDSMovieCloneMap *blayer;
   int nx;
   int ny;
   int nf;
   int buf_index;
   double ts;
   float freq;
   float weight;
   // int numNodes; // Replaced with getNumNodes()
 //  float perf_match;

};

}



#endif /* BIDSSENSORLAYER_HPP_ */
