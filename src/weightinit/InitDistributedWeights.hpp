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
   InitDistributedWeights(HyPerConn * conn);
   virtual ~InitDistributedWeights();

   virtual InitWeightsParams * createNewWeightParams();
   virtual int calcWeights();

protected:
   InitDistributedWeights();
   int initialize_base();
   int initialize(HyPerConn * conn);
};

} /* namespace PV */
#endif /* INITUNIFORMRANDOMWEIGHTS_HPP_ */
