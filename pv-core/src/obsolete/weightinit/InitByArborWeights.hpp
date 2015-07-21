/*
 * InitByArbor.hpp
 *
 *      Author: slundquist 
 */

#ifndef INITBYARBORWEIGHTS_HPP_
#define INITBYARBORWEIGHTS_HPP_

#include "InitWeights.hpp"
//#include "InitUniformRandomWeightsParams.hpp"

namespace PV {

class InitByArborWeights: public PV::InitWeights {
public:
   InitByArborWeights(HyPerConn * conn);
   virtual ~InitByArborWeights();

   virtual int calcWeights(pvdata_t * dataStart, int patchIndex, int arborId);

protected:
   InitByArborWeights();
   int initialize(HyPerConn * conn);

private:
   int initialize_base();
};

} /* namespace PV */
#endif /* INITUNIFORMRANDOMWEIGHTS_HPP_ */
