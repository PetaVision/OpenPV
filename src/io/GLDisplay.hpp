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
   GLDisplay(int * argc, char * argv[], HyPerCol * hc, int numRows, int numCols);
   virtual ~GLDisplay();

   int addDisplay(LayerDataInterface * d);
   int addLayer(HyPerLayer * l);
   int setImage(Image * image);

   void run(float time, float stopTime);

   void drawDisplays();
   int  loadTexture(int id, LayerDataInterface * image);

   // implement LayerProbe interface
   //
   int outputState(float time, HyPerLayer * l);

   void setDelay(float msecs);

private:

   int xTranslate(int index);
   int yTranslate(int index);
   int getTextureId(LayerDataInterface * l);

   HyPerCol * parent;
   Image    * image;
   LayerDataInterface ** displays;

   int numRows;      // number of display rows
   int numCols;      // number of display columns
   int numDisplays;  // current number of displays

   int * textureIds; // texture ids for displays

   float time;
   float stopTime;
   float lastUpdateTime;
};

}

#endif /* GLPROBE_HPP_ */
