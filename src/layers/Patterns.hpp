/*
 * Patterns.hpp
 *
 *  Created on: April 21, 2010
 *      Author: Craig Rasmussen
 */

#ifndef PATTERNS_HPP_
#define PATTERNS_HPP_

#include "Image.hpp"
#include <vector>
namespace PV {

enum PatternType {
  BARS  = 0,
  RECTANGLES  = 1,
  SINEWAVE  = 2,
  COSWAVE  = 3,
  IMPULSE  = 4,
  SINEV  = 5,
  COSV  = 6,
  DROP = 7,
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

typedef struct _Drop{
   int centerX;
   int centerY;
   float speed;
   float radius;
   bool on;
} Drop;

class Patterns : public PV::Image {
public:
   Patterns(const char * name, HyPerCol * hc, PatternType type);
   virtual ~Patterns();
   virtual int updateState(float timef, float dt);

   void setProbMove(float p)     {pMove = p;}
   void setProbSwitch(float p)   {pSwitch = p;}

   void setMinWidth(int w)  {minWidth  = w;}
   void setMaxWidth(int w)  {maxWidth  = w;}
   void setMinHeight(int h) {minHeight = h;}
   void setMaxHeight(int h) {maxHeight = h;}

   virtual int tag();

   int checkpointRead(const char * cpDir, float * timef);
   int checkpointWrite(const char * cpDir);

protected:

   Patterns();
   int initialize(const char * name, HyPerCol * hc, PatternType type);
   virtual int readOffsets() {return PV_SUCCESS;}
   OrientationMode readOrientation();
   int setOrientation(OrientationMode ormode);
   MovementType readMovementType();
   int drawPattern(float val);
   int drawBars(OrientationMode ormode, pvdata_t * buf, int nx, int ny, float val);
   int drawRectangles(float val);
   int drawWaves(float val);
   int drawImpulse();
   int drawDrops();
   int updatePattern(float timef);
   float calcPosition(float pos, int step);

   PatternType type;
   OrientationMode orientation;
   OrientationMode lastOrientation;
   MovementType movementType; //save the type of movement
                              //(random walk, horizontal or vertical drift
                              //or random jumping)

//   int writeImages;  // Base class Image already has member variable writeImages
   int writePosition;     // write positions to input/image-pos.txt
   float position;
   float pSwitch;
   float pMove;
   float movementSpeed; //save a movement speed in pixels/time step
   float positionBound; // The supremum of possible values of position
   int framenumber;

   std::vector <Drop>vDrops;
   float dropSpeed;
   float dropSpeedRandomMax;
   float dropSpeedRandomMin;
   int dropPeriod;
   int dropPeriodRandomMax;
   int dropPeriodRandomMin;
   int nextDropFrame;
   int nextPosChangeFrame;
   int xPos;
   int yPos;
   int onOffFlag;
   int dropPosition;
   int dropPositionRandomMax;
   int dropPositionRandomMin;
   int startFrame;
   int endFrame;

   int minWidth, maxWidth;
   int minHeight, maxHeight;
   int wavelengthVert;
   int wavelengthHoriz;
   float maxVal;
   float minVal;
   char * patternsOutputPath;  // path to output file directory for patterns

   float displayPeriod;   // length of time a frame is displayed
   float nextDisplayTime; // time of next frame
   FILE * patternsFile;

private:
   float rotation;

   int initPatternCntr;
   int initialize_base();
};

}

#endif /* PATTERNS_HPP_ */
