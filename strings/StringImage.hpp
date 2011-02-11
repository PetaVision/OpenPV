/*
 * Patterns.hpp
 *
 *  Created on: April 21, 2010
 *      Author: Craig Rasmussen
 */

#ifndef STRINGIMAGE_HPP_
#define STRINGIMAGE_HPP_

#include <src/layers/Retina.hpp>

namespace PV {

enum PatternType {
  BARS  = 0,
  RECTANGLES  = 1,
};

enum StringMode {left, right};

class StringImage : public PV::Retina {
public:
   StringImage(const char * name, HyPerCol * hc);
   virtual ~StringImage();

   virtual int recvSynapticInput(HyPerConn * conn, PVLayerCube * cube, int neighbor);
   virtual int updateState(float time, float dt);

   void setProbMove(float p)     {pMove = p;}
   void setProbJitter(float p)   {pJit  = p;}

   virtual int tag();

protected:

   int initPattern();

   PatternType type;
   StringMode orientation;
   StringMode lastOrientation;

   int writeImages;
   int position;
   int jitter;

   float pJit;
   float pMove;
};

}

#endif /* STRINGIMAGE_HPP_ */
