/*
 * BIDSSensorLayer.h
 *
 *  Created on: Sep 7, 2012
 *      Author: slundquist
 */

#ifndef BIDSSENSORLAYER_H_
#define BIDSSENSORLAYER_H_

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
   int updateState(double timef, double dt);
protected:
   BIDSSensorLayer();
   float matchFilter(int node_index, int frame_index);
   void writeCSV(std::string fname, int node_index);
//   float perfectMatch();
   float** data;
   BIDSCoords* coords;
   int buf_size;
   float neutral_val;
   BIDSMovieCloneMap *blayer;
   int nx;
   int ny;
   int nf;
   int buf_index;
   double ts;
   float freq;
   float weight;
   int numNodes;
 //  float perf_match;

};

}



#endif /* BIDSSENSORLAYER_H_ */
