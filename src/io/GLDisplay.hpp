/*
 * GLDisplay.hpp
 *
 *  Created on: Jan 9, 2010
 *      Author: Craig Rasmussen
 */

#ifndef GLPROBE_HPP_
#define GLPROBE_HPP_

#include "../columns/HyPerCol.hpp"
#include "../layers/Image.hpp"

namespace PV {

class GLDisplay: public HyPerColDelegate {
public:
   GLDisplay(int * argc, char * argv[], HyPerCol * hc, float msec=0.0f);
   virtual ~GLDisplay();

   void setImage(Image * image);

   void run(float time, float stopTime);

   void drawDisplay();
   int  loadTexture(int id, Image * image);

private:
   HyPerCol * parent;
   Image    * image;
   float time;
   float stopTime;
   float lastUpdateTime;
};

}

#endif /* GLPROBE_HPP_ */
