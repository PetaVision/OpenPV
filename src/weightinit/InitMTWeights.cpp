/*
 * InitMTWeights.cpp
 *
 *  Created on: Oct 25, 2011
 *      Author: kpeterson
 */

#include "InitMTWeights.hpp"

namespace PV {

InitMTWeights::InitMTWeights()
{
   initialize_base();
}

InitMTWeights::~InitMTWeights()
{
   // TODO Auto-generated destructor stub
}

int InitMTWeights::initialize_base() {
   return PV_SUCCESS;
}

InitWeightsParams * InitMTWeights::createNewWeightParams(HyPerConn * callingConn) {
   InitWeightsParams * tempPtr = new InitMTWeightsParams(callingConn);
   return tempPtr;
}

int InitMTWeights::calcWeights(PVPatch * patch, int patchIndex, int arborId,
                                   InitWeightsParams *weightParams) {

   InitMTWeightsParams *weightParamPtr = dynamic_cast<InitMTWeightsParams*>(weightParams);


   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(1);
   }


   weightParamPtr->calcOtherParams(patch, patchIndex);

   calculateMTWeights(patch, weightParamPtr);


   return PV_SUCCESS;

}

/**
 * calculate temporal-spatial gaussian filter for use in optic flow detector
 */
int InitMTWeights::calculateMTWeights(PVPatch * patch, InitMTWeightsParams * weightParamPtr) {
   //load necessary params:
   int nfPatch_tmp = weightParamPtr->getnfPatch_tmp();
   //for MT cells, the patch size must be 1!  We are connecting to the V1 outputs
   //from the cells directly underneath.
   int nyPatch_tmp = weightParamPtr->getnyPatch_tmp();
   assert(nyPatch_tmp==1);
   int nxPatch_tmp = weightParamPtr->getnxPatch_tmp();
   assert(nxPatch_tmp==1);
   //int sx_tmp=weightParamPtr->getsx_tmp();
   //int sy_tmp=weightParamPtr->getsy_tmp();
   int sf_tmp=weightParamPtr->getsf_tmp();

   ChannelType channel = weightParamPtr->getParentConn()->getChannel();

   float speed = weightParamPtr->getSpeed();
   float thetaPre = weightParamPtr->getthPre();
   float v1Speed = weightParamPtr->getV1Speed();

   float v1X, v1Y, v1T;
   calculateVector(thetaPre, v1Speed, v1X, v1Y, v1T);

   pvdata_t * w_tmp = patch->data;



   // loop over all post-synaptic cells in temporary patch
   for (int fPost = 0; fPost < nfPatch_tmp; fPost++) {
      float thPost = weightParamPtr->calcThPost(fPost);

      float mtX, mtY, mtT;

      calculateMTPlane(thPost, speed, mtX, mtY, mtT);

      float distance = calcDist(v1X, v1Y, v1T, mtX, mtY, mtT);
      //printf("distance %f\n",distance);
      int index = fPost * sf_tmp;
      w_tmp[index] = 0;
      if((channel == CHANNEL_EXC)&&(distance>-0.01)&&(distance<0.01)) {
         w_tmp[index] = 1;
      }
      else if(channel == CHANNEL_EXC) {
         w_tmp[index] = 0;
      }
      else if(channel == CHANNEL_INH) {
         w_tmp[index] = distance;
      }
      else {
         assert((channel == CHANNEL_INH)||(channel == CHANNEL_EXC));
      }
   }

   return PV_SUCCESS;
}

int InitMTWeights::calculateVector(float theta, float speed, float &x, float &y, float &t) {
//   if(speed==0) {
//      t=0;
//      x=cos(theta);
//      y=sin(theta);
//   }
//   else {
//      t=1;
//      x=speed*cos(theta);
//      y=speed*sin(theta);
//   }

   t=speed;
   x=cos(theta);
   y=sin(theta);

//   printf("1st vector x %f\n",x);
//   printf("1st vector y %f\n",y);
//   printf("1st vector t %f\n",t);

   return PV_SUCCESS;
}

int InitMTWeights::calculateMTPlane(float theta, float speed, float &x, float &y, float &t) {
   float p1X, p1Y, p1T;
   calculateVector(theta, speed, p1X, p1Y, p1T);

   float p2X, p2Y, p2T;
   calculate2ndVector(p1X, p1Y, p1T, p2X, p2Y, p2T);

   x=p1Y*p2T-p1T*p2Y;
   y=-(p1X*p2T-p1T*p2X);
   t=p1X*p2Y-p1Y*p2X;

   return PV_SUCCESS;
}

int InitMTWeights::calculate2ndVector(float p1x, float p1y, float p1t, float &p2x, float &p2y, float &p2t) {
//   p2x=p1x+p1y;
//   p2y=p1y-p1x;
//   p2t=p1t;

   float vx=-p1x;
   float vy=-p1y;
   //float s2=p1t;

   float u2x=-vy;
   float u2y=vx;
   float u2t=0;
//   printf("2nd vector u2x %f\n",u2x);
//   printf("2nd vector u2y %f\n",u2y);
//   printf("2nd vector u2t %f\n",u2t);

   p2x=p1x+u2x;
   p2y=p1y+u2y;
   p2t=p1t+u2t;
//   printf("3rd vector p2x %f\n",p2x);
//   printf("3rd vector p2y %f\n",p2y);
//   printf("3rd vector p2t %f\n",p2t);
   return PV_SUCCESS;
}

float InitMTWeights::calcDist(float v1x, float v1y, float v1t, float mtx, float mty, float mtt) {
//   float x=v1x*mtx;
//   float y=v1y*mty;
//   float t=v1t*mtt;
//   float tot=x+y+t;
//   float mtxx=mtx*mtx;
//   float mtyy=mty*mty;
//   float mttt=mtt*mtt;
//   float tot2=mtxx+mtyy+mttt;
//   float den=sqrt(tot2);
//   float ans=tot/den;
//   float ans2=fabs(ans);

   return fabs((v1x*mtx + v1y*mty + v1t*mtt)/sqrt(mtx*mtx + mty*mty + mtt*mtt));
}


} /* namespace PV */
