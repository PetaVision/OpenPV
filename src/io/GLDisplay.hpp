/*
 * GLDisplay.hpp
 *
 *  Created on: Jan 9, 2010
 *      Author: Craig Rasmussen
 */

#ifndef GLPROBE_HPP_
#define GLPROBE_HPP_

#include "../columns/HyPerCol.hpp"
#include "../layers/HyPerLayer.hpp"
#include "../layers/Image.hpp"

namespace PV {

class GLDisplay: public HyPerColRunDelegate, public LayerProbe {
public:
   GLDisplay(int * argc, char * argv[], HyPerCol * hc, float msec=0.0f);
   virtual ~GLDisplay();

   int addLayer(HyPerLayer * l);
   void setImage(Image * image);

   void run(float time, float stopTime);

   void drawDisplay();
   int  loadTexture(int id, LayerDataInterface * image);

   // implement LayerProbe interface
   //
   int outputState(float time, HyPerLayer * l);

private:
   HyPerCol   * parent;
   Image      * image;
   HyPerLayer * layer;

   int layerTexId;

   float time;
   float stopTime;
   float lastUpdateTime;
};

}

#endif /* GLPROBE_HPP_ */
