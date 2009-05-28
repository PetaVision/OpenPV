
/*
 * Retina1D.hpp
 *
 *  Created on: Apr 25, 2009
 *      Author: gkenyon
 */

#ifndef RETINA1D_HPP_
#define RETINA1D_HPP_

#include <src/layers/Retina.hpp>

namespace PV {

class Retina1D: public PV::Retina {
public:
   Retina1D();
   Retina1D(const char * name, HyPerCol * hc);
   virtual ~Retina1D();

   virtual int createImage(pvdata_t * buf);
   virtual int createRandomImage(pvdata_t * buf);
   virtual int spike(float time, float dt, float prob, int * start);
   virtual int updateState(float time, float dt);

protected:
   pvdata_t * targ1D;
};

}

#endif /* RETINA1D_HPP_ */
