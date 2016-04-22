/*
 * CreateMovies.cpp
 *
 *  Created on: Dec. 12, 2009
 *      Author: Wentao
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../columns/InterColComm.hpp"

#include "CreateMovies.hpp"

#define Pi 3.14159265

#define DEBUG_OUTPUTIMAGES
#ifdef DEBUG_OUTPUTIMAGES
int numFrame = 0;
int MaxNumberFrames = 20;
#endif


CreateMovies_Params DefaultCMParams={
      32,//nx
      32,//ny
      255,//foregroundval
      0,//backgroudval
      1,//isgray
      0,//rotateangle
      0,//centerx
      0,//centery
      8,//period
      3,//linewidth
      2,//vx
      2,//vy
      5,//vr
      -1,//isshiftx
      0,//isshifty
      0,//isrotate
};

namespace PV {

CreateMovies::CreateMovies() {
   initialize_base();
}

CreateMovies::CreateMovies(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
   // initialize_Movies(hc);
}

CreateMovies::~CreateMovies() {
   free(CMParams);
}

int CreateMovies::initialize_base() {
   data = NULL;
   CMParams = NULL;
   return PV_SUCCESS;
}

int CreateMovies::initialize(const char * name, HyPerCol * hc) {
   Image::initialize(name, hc);

   PVParams * pvparams = hc->parameters();
   // setMovieParams(pvparams, &DefaultCMParams); // Duplicated below

   lastDisplayTime = hc->simulationTime();
   nextDisplayTime = hc->simulationTime() + displayPeriod;

   flagx = 1;
   flagy = 1;
   flagr = 1;

#ifdef DEBUG_OUTPUTIMAGES
   double T = parent->simulationTime() ;
   char title[1000];
   ::sprintf(title,"output/images/%05d.tif",(int)T);
   write(title);
#endif

   return PV_SUCCESS;
}

int CreateMovies::allocateDataStructures() {
   int status = Image::allocateDataStructures();

   const PVLayerLoc * loc = getLayerLoc();
   free(data);
   size_t dn = loc->nf * (loc->nx + loc->halo.lt + loc->halo.rt)
                                  * (loc->ny + loc->halo.dn + loc->halo.up) * sizeof(pvdata_t);
   data = (pvdata_t *) malloc(dn);
   assert(data != NULL);
   memset((pvdata_t *)data, (int)(CMParams->backgroundval), dn);
   Transform(0, 0, 0);

   return status;
}

int CreateMovies::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = Image::ioParamsFillGroup(ioFlag);
   parent->ioParamValue(ioFlag, name, "nx", &CMParams->nx, CMParams->nx);
   parent->ioParamValue(ioFlag, name, "ny", &CMParams->ny, CMParams->ny);
   parent->ioParamValue(ioFlag, name, "foregroundval", &CMParams->foregroundval, CMParams->foregroundval);
   parent->ioParamValue(ioFlag, name, "backgroundval", &CMParams->backgroundval, CMParams->backgroundval);
   parent->ioParamValue(ioFlag, name, "isgray", &CMParams->isgray, CMParams->isgray);
   parent->ioParamValue(ioFlag, name, "rotateangle", &CMParams->rotateangle, CMParams->rotateangle);
   parent->ioParamValue(ioFlag, name, "centerx", &CMParams->centerx, CMParams->centerx);
   parent->ioParamValue(ioFlag, name, "centery", &CMParams->centery, CMParams->centery);
   parent->ioParamValue(ioFlag, name, "period", &CMParams->period, CMParams->period);
   parent->ioParamValue(ioFlag, name, "linewidth", &CMParams->linewidth, CMParams->linewidth);
   parent->ioParamValue(ioFlag, name, "vx", &CMParams->vx, CMParams->vx);
   parent->ioParamValue(ioFlag, name, "vy", &CMParams->vy, CMParams->vy);
   parent->ioParamValue(ioFlag, name, "vr", &CMParams->vr, CMParams->vr);
   parent->ioParamValue(ioFlag, name, "isshiftx", &CMParams->isshiftx, CMParams->isshiftx);
   parent->ioParamValue(ioFlag, name, "isshifty", &CMParams->isshifty, CMParams->isshifty);
   parent->ioParamValue(ioFlag, name, "isrotate", &CMParams->isrotate, CMParams->isrotate);
   parent->ioParamValue(ioFlag, name, "displayPeriod", &displayPeriod, 20.0f);
   return status;
}

void CreateMovies::ioParam_imagePath(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      filename = NULL;
      parent->parameters()->handleUnnecessaryStringParameter(name, "imageList", NULL);
   }
}


int CreateMovies::Rotate(const float DAngle, const int centerx, const int centery)
{
   const PVLayerLoc * loc = getLayerLoc();

   CreateMovies_Params * param = (CreateMovies_Params *)CMParams;

   int Nx = loc->nx + loc->halo.lt + loc->halo.rt;
   int Ny = loc->ny + loc->halo.dn + loc->halo.up;

   param->rotateangle+=DAngle;
   if (param->rotateangle>=180){
      param->rotateangle=((int)(param->rotateangle)) % 180 + param->rotateangle-(int)param->rotateangle;
   }

   if (param->rotateangle<=-180){
      param->rotateangle=((int)(param->rotateangle)) % 180 + param->rotateangle-(int)param->rotateangle;
   }

   float Agl = Pi*param->rotateangle/180.0;
   int cx = (int)(Nx/2.0), cy = (int)(Ny/2.0);
   int period = param->period;
   int linewidth = param->linewidth;
   pvdata_t foregroundval = param->foregroundval;
   pvdata_t backgroundval = param->backgroundval;

   size_t dn=loc->nf * (loc->nx + loc->halo.lt + loc->halo.rt)
                           * (loc->ny + loc->halo.dn + loc->halo.up) * sizeof(pvdata_t);
   if (data == NULL){
      data = (pvdata_t *) malloc(dn);
   }
   assert(data != NULL);

   memset((pvdata_t *)data, (int)backgroundval, dn);

   float cs = ::cos((double)Agl),  sn = ::sin((double)Agl);
   int i,j,i1;


   float a = centerx*cs + centery*sn;
   //float b = centery*cs - centerx*sn;

   int cx1 = Nx-cx;
   int cy1 = Ny-cy;
   float Pd = 2*Pi/period;
   float fb = (foregroundval-backgroundval)/2;

   for(i=-cx; i<cx1; i++){
      for(j=-cy; j<cy1; j++){
         i1=(int)(i*cs + j*sn-a);//(int)(i*cs+0.5 + j*sn-a + 0.5);
         //j1=j*cs - i*sn-b;
         if (param->isgray){
            data[i+cx+(j+cy)*Nx] = backgroundval + (1+::cos(Pd*i1))*fb;
         }
         else{
            int t=i1%period;
            if ((t>=0 && t< linewidth)||(t<0 && t+period < linewidth)){
               data[i+cx+(j+cy)*Nx] = foregroundval;
            }
         }

      }
   }

   return 0;
}

int CreateMovies::Transform(const float DAngle,const int Dx,const int Dy)
{
   const PVLayerLoc * loc = getLayerLoc();

   CreateMovies_Params * param = CMParams;

   int Nx = loc->nx + loc->halo.lt + loc->halo.rt;
   int Ny = loc->ny + loc->halo.dn + loc->halo.up;

   param->centerx +=  Dx;
   param->centerx  =  (int)param->centerx % (int)(10*Nx);
   param->centery +=  Dy;
   param->centery  =  (int)param->centery % (int)(10*Ny);

   return Rotate(DAngle, param->centerx, param->centery);
}

bool CreateMovies::updateImage(float time, float dt){


   if(time - lastDisplayTime < dt/2) {
      return true;
   }

   if (time < nextDisplayTime) {
      return false;
   }

   lastDisplayTime = time;

   nextDisplayTime += displayPeriod;

   CreateImages();

#ifdef DEBUG_OUTPUTIMAGES
   if (numFrame < MaxNumberFrames){
      int T = (int)(nextDisplayTime - displayPeriod);
      char title[1000];
      ::sprintf(title,"output/images/%05d.tif",T);
      write(title);
      numFrame++;
   }
#endif

   return true;
}



bool CreateMovies::CreateImages(){

   //CreateMovies_Params *param = (CreateMovies_Params *)CMParams;

   int isshiftx = CMParams->isshiftx;
   int isshifty = CMParams->isshifty;
   int isrotate = CMParams->isrotate;
   int flag;

   // Create different pattern image sequences
   while(1){

      if (isshiftx != 0 && flagx != 0){
         if (isshiftx<0)
            Transform(0,CMParams->vx,0);
         else{
            flag = flagx % (isshiftx+1);
            if (flag != 0){
               Transform(0,CMParams->vx,0);
               flagy = 0;
               flagr = 0;
               flagx++;
               break;
            }
            else{
               flagx = 0;
               flagy = 1;
               flagr = 1;
               continue;
            }

         }
         break;
      }

      if (isshifty != 0 && flagy != 0){
         if (isshifty<0)
            Transform(0,0,CMParams->vy);
         else{
            flag = flagy % (isshifty+1);
            if (flag != 0){
               Transform(0,0,CMParams->vy);
               flagx = 0;
               flagr = 0;
               flagy++;
               break;
            }
            else{
               flagx = 1;
               flagy = 0;
               flagr = 1;
               continue;
            }

         }
         break;
      }

      if (isrotate != 0 && flagr != 0){
         if (isrotate<0)
            Transform(CMParams->vr,0,0);
         else{
            flag = flagr % (isrotate+1);
            if (flag != 0){
               Transform(CMParams->vr,0,0);
               flagx = 0;
               flagy = 0;
               flagr++;
               break;
            }
            else{
               flagx = 1;
               flagy = 1;
               flagr = 0;
               continue;
            }

         }
         break;
      }

      break;
   }//while
      return true;
}

}//namespace PV





