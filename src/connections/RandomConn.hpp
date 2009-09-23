/*
 * RandomConn.hpp
 *
 *  Created on: Apr 27, 2009
 *      Author: rasmussn
 */

#ifndef RANDOMCONN_HPP_
#define RANDOMCONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

enum RandDistType {
   UNDEFINED = 0,
   UNIFORM = 1,
   GAUSSIAN = 2
};

class RandomConn: public PV::HyPerConn {
public:
   RandomConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
              HyPerLayer * post, int channel);

   RandomConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
              HyPerLayer * post, int channel, RandDistType distrib);

   virtual int initializeRandomWeights(int seed);
   int initializeUniformWeights(int seed);
   int uniformWeights(PVPatch * wp, float wMin, float wMax, int seed);
   int initializeGaussianWeights(int seed);
   int gaussianWeights(PVPatch *wp, float mean, float stdev, int seed);
   int gaussianWeightsMA(PVPatch *wp, float mean, float stdev, long *);
   float randgauss(float mean, float stdev);
   float randgaussMA(float mean, float stdev, long *);
private:
   float          wMin;
   RandDistType   randDistType; // the type of distribution
   float          wGaussMean;   // mean of the Gaussian distribution
   float          wGaussStdev;  // std deviation of the Gaussian distribution
   long           idum;

   float ranf();
   float ran1(long *);
};

}

#endif /* RANDOMCONN_HPP_ */
