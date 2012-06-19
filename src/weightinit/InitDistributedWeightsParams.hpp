/*
 * InitDistributedWeightsParams.hpp
 *
 *  Created on: Jun 18, 2012
 *      Author: bnowers
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

   //get-set methods:
   inline float getWMin()        {return wMin;}
   inline float getWMax()        {return wMax;}
   inline float getNumNodes()    {return numNodes;}

protected:
   virtual int initialize_base();
   int initialize(HyPerConn * parentConn);


private:
   float wMin;
   float wMax;
   float numNodes;
};

} /* namespace PV */
#endif /* INITDISTRIBUTEDWEIGHTSPARAMS_HPP_ */
