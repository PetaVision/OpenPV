/*
 * CliqueConn.hpp
 *
 *  Created on: Sep 8, 2011
 *      Author: gkenyon
 */

#ifndef CLIQUECONN_HPP_
#define CLIQUECONN_HPP_

#include "KernelConn.hpp"

namespace PV {

class CliqueConn: public KernelConn {
public:

   CliqueConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
              const char * filename, InitWeights *weightInit);
   virtual int update_dW(int axonId);
   virtual int updateState(float time, float dt);
   virtual int updateWeights(int arbor);
   //virtual int normalizeWeights(PVPatch ** patches, int numPatches, int arborId);

protected:
   int cliqueSize; // number of presynaptic cells in clique (traditional ANN uses 1)
   int initialize(const char * name, HyPerCol * hc, HyPerLayer * pre,
         HyPerLayer * post, const char * filename,
         InitWeights *weightInit);

private:
   int initialize_base();

}; // class CliqueConn

} /* namespace PV */
#endif /* CLIQUECONN_HPP_ */
