/*
 * CPTestInputLayer.hpp
 *
 *  Created on: Nov 10, 2011
 *      Author: pschultz
 */

#ifndef CPTESTINPUTLAYER_HPP_
#define CPTESTINPUTLAYER_HPP_

#include "../PetaVision/src/layers/ANNLayer.hpp"

namespace PV {

class CPTestInputLayer : public ANNLayer {

public:
   CPTestInputLayer(const char * name, HyPerCol * hc);
   virtual ~CPTestInputLayer();
   virtual int updateV();

protected:
   int initialize();
   virtual int initializeV(bool restart_flag);

}; // end class CPTestInputLayer

}  // end of namespace PV block


#endif /* CPTESTINPUTLAYER_HPP_ */
