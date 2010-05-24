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

enum PatternType {
  BARS  = 0,
  RECTANGLES  = 1,
};

enum PatternMode {vertical, horizontal};

class Patterns : public PV::Image {
public:
   Patterns(const char * name, HyPerCol * hc, PatternType type);
   virtual ~Patterns();
   virtual int updateState(float time, float dt);

   void setProbMove(float p) {pMove = p;}
   void setProbSwitch(float p) {pSwitch = p;}

   virtual int tag();

protected:

   int initPattern(float val);
   int calcPosition(int pos, int step);

   PatternType type;
   PatternMode orientation;
   PatternMode lastOrientation;

   int   writeImages;
   int position;
   int lastPosition;
   int prefPosition;
   float pSwitch;
   float pMove;
};

}

#endif /* PATTERNS_HPP_ */
