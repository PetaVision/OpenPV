/*
 * RetinaGrating.hpp
 *
 *  Created on: Jun 7, 2009
 *      Author: rasmussn
 */

#ifndef RETINAGRATING_HPP_
#define RETINAGRATING_HPP_

#include <src/layers/Retina.hpp>

namespace PV {

class RetinaGrating: public PV::Retina {
public:
   RetinaGrating();
   RetinaGrating(const char * name, HyPerCol * hc, int opt);

   virtual int createImage(pvdata_t * buf, int opt);
   virtual int updateState(float time, float dt);
};

}

#endif /* RETINAGRATING_HPP_ */
