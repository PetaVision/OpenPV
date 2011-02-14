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

class StringImage : public PV::Retina {
public:
   StringImage(const char * name, HyPerCol * hc);
   virtual ~StringImage();

   virtual int recvSynapticInput(HyPerConn * conn, PVLayerCube * cube, int neighbor);
   virtual int updateState(float time, float dt);
   virtual int outputState(float time, float dt);


   void setProbMove(float p)     {pMove = p;}
   void setProbJitter(float p)   {pJit  = p;}

   virtual int tag();

protected:

   int initializeString();
   int shiftString();
   int updateLayerData();
   int updateString();

   int * string;
   int   strWidth;
   int   patternWidth;  // width of substring before pattern repeats
   int   phase;         // phase/location of character within repeated substring
   int   jitter;        // position of jitter (left==0/right==1)

   int writeImages;

   float pJit;
   float pMove;
};

}

#endif /* STRINGIMAGE_HPP_ */
