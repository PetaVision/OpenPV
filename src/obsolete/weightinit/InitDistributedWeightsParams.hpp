/*
 * InitDistributedWeightsParams.hpp
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

#ifndef INITDISTRIBUTEDWEIGHTSPARAMS_HPP_
#define INITDISTRIBUTEDWEIGHTSPARAMS_HPP_

#include "InitWeightsParams.hpp"

namespace PV {

class InitDistributedWeightsParams: public PV::InitWeightsParams {
public:
   InitDistributedWeightsParams();
   InitDistributedWeightsParams(HyPerConn * parentConn);
   virtual ~InitDistributedWeightsParams();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   //get-set methods:
   inline float getNumNodes()    {return numNodes;}

protected:
   int initialize_base();
   int initialize(HyPerConn * parentConn);


private:
   float numNodes;
};

} /* namespace PV */
#endif /* INITDISTRIBUTEDWEIGHTSPARAMS_HPP_ */
