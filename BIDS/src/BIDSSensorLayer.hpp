/*
 * BIDSSensorLayer.hpp
 *
 *  Created on: Sep 7, 2012
 *      Author: slundquist
 */

#ifndef BIDSSENSORLAYER_HPP_
#define BIDSSENSORLAYER_HPP_

#include <layers/HyPerLayer.hpp>
#include "BIDSMovieCloneMap.hpp"
#include <math.h>
//For testing
#include <iostream>
#include <fstream>

namespace PV{

class BIDSSensorLayer : public Image{
public:
   BIDSSensorLayer(const char * name, HyPerCol * hc);
   virtual ~BIDSSensorLayer();
   int initialize_base();
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
   //bool needUpdate(double time, double dt);
   int updateState(double timef, double dt);
protected:
   BIDSSensorLayer();
   int initialize(const char * name, HyPerCol * hc);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_frequency(enum ParamsIOFlag ioFlag);
   virtual void ioParam_ts_per_period(enum ParamsIOFlag ioFlag);
   virtual void ioParam_buffer_size(enum ParamsIOFlag ioFlag);
   virtual void ioParam_neutral_val(enum ParamsIOFlag ioFlag);
   virtual void ioParam_weight(enum ParamsIOFlag ioFlag);
   virtual void ioParam_jitterSource(enum ParamsIOFlag ioFlag);
   float matchFilter(int node_index, int frame_index);
   void writeCSV(std::string fname, int node_index);
   BIDSCoords * getCoords() {return blayer->getCoords();}
   int getNumNodes() {return blayer->getNumNodes();}
   virtual double getDeltaUpdateTime();
   //   float perfectMatch();

protected:
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
