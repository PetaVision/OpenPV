/*
 * Gratings.hpp
 *
 *  Created on: Oct 23, 2009
 *      Author: travel
 */

#ifndef BARS_HPP_
#define BARS_HPP_

#include "Image.hpp"

namespace PV {

enum or_modes {vertical, horizontal};

class Bars : public PV::Image {
public:
   Bars(const char * name, HyPerCol * hc);
   virtual ~Bars();
   void setProbSwitch(float p) {pSwitch = p;}
   void setProbMove(float p) {pMove = p;}
   virtual bool updateImage(float time, float dt);
   virtual int  clearImage();

protected:

   void calcPosition(float step);
   int   writeImages;
   float position;
   float lastPosition;
   or_modes orientation;
   or_modes lastOrientation;
   float pSwitch;
   float pMove;
   int random_walk;
   int move_forward;
   int move_backward;
   int random_jump;
};

}

#endif /* BARS_HPP_ */
