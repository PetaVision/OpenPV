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

   void setProbMove(float p)     {pMove = p;}
   void setProbSwitch(float p)   {pSwitch = p;}

   void setMinWidth(float w)  {minWidth  = w;}
   void setMaxWidth(float w)  {maxWidth  = w;}
   void setMinHeight(float h) {minHeight = h;}
   void setMaxHeight(float h) {maxHeight = h;}

   virtual int tag();

protected:

   int initPattern(float val);
   int initPattern(float val,float time);
   int clearPattern(float val);
   int calcPosition(int pos, int step);

   PatternType type;
   PatternMode orientation;
   PatternMode lastOrientation;

   int writeImages;
   int position;
   int lastPosition;
   int prefPosition;
   float pSwitch;
   float pMove;

   float minWidth, maxWidth;
   float minHeight, maxHeight;
};

}

#endif /* PATTERNS_HPP_ */
