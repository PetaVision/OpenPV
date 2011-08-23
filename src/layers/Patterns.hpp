/*
 * Patterns.hpp
 *
 *  Created on: April 21, 2010
 *      Author: Craig Rasmussen
 */

#ifndef PATTERNS_HPP_
#define PATTERNS_HPP_

#include "Image.hpp"

namespace PV {

enum PatternType {
  BARS  = 0,
  RECTANGLES  = 1,
};

enum PatternMode {
   vertical = 0,
   horizontal = 1,
};

enum MovementType {
   RANDOMWALK = 0,
   MOVEFORWARD = 1,
   MOVEBACKWARD = 2,
   RANDOMJUMP = 3,
};

class Patterns : public PV::Image {
public:
   Patterns(const char * name, HyPerCol * hc, PatternType type);
   virtual ~Patterns();
   virtual int updateState(float time, float dt);

   void setProbMove(float p)     {pMove = p;}
   void setProbSwitch(float p)   {pSwitch = p;}

   void setMinWidth(int w)  {minWidth  = w;}
   void setMaxWidth(int w)  {maxWidth  = w;}
   void setMinHeight(int h) {minHeight = h;}
   void setMaxHeight(int h) {maxHeight = h;}

   virtual int tag();

protected:

   int initPattern(float val);
   int calcPosition(int pos, int step);

   PatternType type;
   PatternMode orientation;
   PatternMode lastOrientation;
   MovementType movementType; //save the type of movement
                              //(random walk, horizontal or vertical drift
                              //or random jumping)

   int writeImages;
   int position;
   int lastPosition;
   int prefPosition;
   float pSwitch;
   float pMove;
   int movementSpeed; //save a movement speed in pixels/time step


   int minWidth, maxWidth;
   int minHeight, maxHeight;
};

}

#endif /* PATTERNS_HPP_ */
