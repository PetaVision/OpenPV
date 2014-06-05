/*
 * CliqueConn.hpp
 *
 *  Created on: Sep 8, 2011
 *      Author: gkenyon
 */

#ifndef CLIQUECONN_HPP_
#define CLIQUECONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class CliqueConn: public HyPerConn {
public:

   CliqueConn(const char * name, HyPerCol * hc);
   virtual int update_dW(int axonId);
   virtual int updateState(double time, double dt);
   virtual int updateWeights(int arbor);
   //virtual int normalizeWeights(PVPatch ** patches, int numPatches, int arborId);

protected:
   int cliqueSize; // number of presynaptic cells in clique (traditional ANN uses 1)
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag);
   virtual void ioParam_cliqueSize(enum ParamsIOFlag ioFlag);

private:
   int initialize_base();

}; // class CliqueConn

} /* namespace PV */
#endif /* CLIQUECONN_HPP_ */
