/*
 * Gratings.hpp
 *
 *  Created on: Oct 23, 2009
 *      Author: travel
 */

#ifndef GRATINGS_HPP_
#define GRATINGS_HPP_

#include "Image.hpp"

namespace PV {

class Gratings : public PV::Image {
public:
   Gratings(const char * name, HyPerCol * hc);
   virtual ~Gratings();
   void setProbMove(float p) {pMove = p;}
   virtual bool updateImage(float time, float dt);

protected:

   float calcPhase(float time, float dt);

   float phase; // lastPhase inherited from Image
   float period;
   float pMove;

};

}

#endif /* GRATINGS_HPP_ */
