/*
 * InitDistributedWeights.hpp
 *
 *  Created on: Jun 18, 2012
 *      Author: bnowers
 *
 *  NOTES: This weight initialization class can ONLY be used in a HyPer Connection. It will
 *  not work with a Kernel Connection. The purpose of this class is to sparsely fill the patch
 *  matrix with a specified amount of neurons (nodes) that are randomly distributed throughout
 *  the matrix. To specify the number of nodes, add a numNodes parameter to the HyPerConn you
 *  wish to use in the params file.
 */

#ifndef INITDISTRIBUTEDWEIGHTS_HPP_
#define INITDISTRIBUTEDWEIGHTS_HPP_

#include "InitWeights.hpp"
#include "InitDistributedWeightsParams.hpp"

namespace PV {

class InitDistributedWeights: public PV::InitWeights {
public:
   InitDistributedWeights();
//   InitUniformRandomWeights(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
//         ChannelType channel);
   virtual ~InitDistributedWeights();
   virtual int initializeWeights(PVPatch *** patches, pvdata_t ** dataStart, int numPatches, const char * filename, HyPerConn * callingConn, double * timef=NULL);

   virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);

protected:
   virtual int initialize_base();
//   int initialize(const char * name, HyPerCol * hc,
//                  HyPerLayer * pre, HyPerLayer * post,
//                  ChannelType channel);


};

} /* namespace PV */
#endif /* INITUNIFORMRANDOMWEIGHTS_HPP_ */
