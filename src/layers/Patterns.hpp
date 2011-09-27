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
  SINEWAVE  = 2,
};

enum OrientationMode {
   horizontal = 0,
   vertical = 1,
   mixed = 2,
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

   int initializePatterns(const char * name, HyPerCol * hc, PatternType type);
   int initPattern(float val);
   float calcPosition(float pos, int step);

   PatternType type;
   OrientationMode orientation;
   OrientationMode lastOrientation;
   MovementType movementType; //save the type of movement
                              //(random walk, horizontal or vertical drift
                              //or random jumping)

   int writeImages;
   int writePosition;     // If using jitter, write positions to input/image-pos.txt
   float position;
   int lastPosition;
   int prefPosition;
   float pSwitch;
   float pMove;
   float movementSpeed; //save a movement speed in pixels/time step


   int minWidth, maxWidth;
   int minHeight, maxHeight;
   char * patternsOutputPath;  // path to output file directory for patterns

};

}

#endif /* PATTERNS_HPP_ */
