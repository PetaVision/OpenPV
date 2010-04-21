/*
 * Patterns.hpp
 *
 *  Created on: April 21, 2010
 *      Author: Craig Rasmussen
 */

#ifndef PATTERNS_HPP_
#define PATTERNS_HPP_

#include <src/layers/Image.hpp>

namespace PV {

enum or_modes {vertical, horizontal};

class Patterns : public PV::Image {
public:
   Patterns(const char * name, HyPerCol * hc);
   virtual ~Patterns();
   void setProbSwitch(float p) {pSwitch = p;}
   void setProbMove(float p) {pMove = p;}
   virtual bool updateImage(float time, float dt);
   virtual int tag();

protected:

   int initPattern(float val);
   int calcPosition(int pos, int step);
   int   writeImages;
   int position;
   int lastPosition;
   int prefPosition;
   or_modes orientation;
   or_modes lastOrientation;
   float pSwitch;
   float pMove;
};

}

#endif /* PATTERNS_HPP_ */
