/*
 * CreateMovies.hpp
 *
 *  Created on: Dec. 12, 2009
 *      Author: Wentao
 */

#ifndef CREATEMOVIES_HPP_
#define CREATEMOVIES_HPP_

#include "Image.hpp"
#include "../layers/HyPerLayer.hpp"

typedef struct CreateMovies_Params_ {

   // base parameter for images
   int nx;  //width
   int ny;  //height
   pvdata_t foregroundval;
   pvdata_t backgroundval;
   bool isgray;

   //transformation parameter
   float rotateangle;//unit: degree
   int centerx;//unit: pixel
   int centery;//unit: pixel

   //image patterns
   int period;//unit: pixel
   int linewidth; //unit: pixel, for isgray = 0;

   //moving velocity
   int vx;//unit: pixel
   int vy;//unit: pixel
   float vr;//unit: degree

   //moving patterns
   int isshiftx;
   int isshifty;
   int isrotate;

}CreateMovies_Params;


namespace PV {

class CreateMovies: public PV::Image {
public:
   CreateMovies(const char * name, HyPerCol * hc);
   virtual ~CreateMovies();
   virtual int allocateDataStructures();

   virtual int Rotate(const float DAngle, const int centerx = 0, const int centery = 0);
   virtual int Shift(const int Dx,const int Dy = 0)  {return Transform(0,Dx,Dy);}
   virtual int Transform(const float DAngle,const int Dx=0,const int Dy=0);

   float getLastDisplayTime(){return lastDisplayTime;}
   float getNextDisplayTime(){return nextDisplayTime;}

   //Image never updates, so needUpdate always returns false
   //Overwriting Image's needUpdate to always update
   //TODO see when this layer actually needs to update
   virtual bool needUpdate(float time, float dt){return true;}

   //TODO Pretty sure this function isn't being called from anywhere
   virtual bool updateImage(float time, float dt);
   virtual bool CreateImages();

protected:
   CreateMovies();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_imagePath(enum ParamsIOFlag ioFlag);
private:
   int initialize_base();

protected:
   float displayPeriod;     // length of time a frame is displayed
   float lastDisplayTime;   //
   float nextDisplayTime;   // time of next frame
   CreateMovies_Params * CMParams;
   int flagx, flagy, flagr;
};

}//namespace PV


#ifdef __cplusplus
extern "C"
{
#endif

#ifdef __cplusplus
}
#endif

#endif /* CREATEMOVIES_HPP_ */
