/*
 * InitWeightsParams.hpp
 *
 *  Created on: Aug 10, 2011
 *      Author: kpeterson
 */

#ifndef INITWEIGHTSPARAMS_HPP_
#define INITWEIGHTSPARAMS_HPP_

#include "../include/pv_common.h"
#include "../include/pv_types.h"
#include "../io/PVParams.hpp"
#include "../layers/HyPerLayer.hpp"
#include "../connections/HyPerConn.hpp"
#include <stdlib.h>
#include <string.h>

namespace PV {
class HyPerConn;

class InitWeightsParams {
public:
   InitWeightsParams();
   InitWeightsParams(HyPerConn * pConn);
   virtual ~InitWeightsParams();

   //get-set methods:
   inline const char * getName()                       {return name;}
   inline void setName(const char * name)    {free(this->name); this->name = strdup(name);}
   inline HyPerCol * getParent()                     {return parent;}
   inline HyPerLayer * getPre()                        {return pre;}
   inline HyPerLayer * getPost()                       {return post;}
   inline HyPerConn * getParentConn()                 {return parentConn;}
   inline ChannelType getChannel()                 {return channel;}

   //virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);
   virtual float calcDthPre();
   virtual float calcTh0Pre(float dthPre);
   float calcThPost(int fPost);
   float calcYDelta(int jPost);
   float calcXDelta(int iPost);
   bool checkTheta(float thPost);

   //get/set:
   int getnfPatch_tmp(); //       {
      //int nf= parentConn->fPatchSize();
//      return 0;
//   }
   int getnyPatch_tmp();//        {return parentConn->yPatchSize();}
   int getnxPatch_tmp();    //    {return 0; } //parentConn->xPatchSize();}
   int getPatchSize_tmp();//      {return 0; } //parentConn->fPatchSize()*
         //parentConn->xPatchSize()*parentConn->yPatchSize();}
   int getsx_tmp();//        {return 0; } //parentConn->xPatchStride();}
   int getsy_tmp();//        {return 0; } //parentConn->yPatchStride();}
   int getsf_tmp();//        {return 0; } //parentConn->fPatchStride();}
   inline float getthPre()        {return thPre;}
   inline int getFPre()        {return fPre;}

protected:
   virtual int initialize_base();
   int initialize(HyPerConn * pConn);


   char * name; //this is actually the Connection name
   HyPerLayer     * pre;
   HyPerLayer     * post;
   HyPerCol       * parent;
   HyPerConn      * parentConn;
   ChannelType channel;    // which channel of the post to update (e.g. inhibit)

   void getcheckdimensionsandstrides();
   int kernelIndexCalculations(int patchIndex);
   void calculateThetas(int kfPre_tmp, int patchIndex);

   //more get/set
   inline float getxDistHeadPreUnits()        {return xDistHeadPreUnits;}
   inline float getyDistHeadPreUnits()        {return yDistHeadPreUnits;}
   inline float getdyPost()        {return dyPost;}
   inline float getdxPost()        {return dxPost;}
   inline int getNoPost()        {return noPost;}
   inline int getNoPre()        {return noPre;}
   inline float getThetaMax()        {return thetaMax;}
   inline float getDeltaThetaMax()        {return deltaThetaMax;}
   inline float getDeltaTheta()        {return deltaTheta;}
   inline float getRotate()        {return rotate;}
   inline void setThetaMax(float thetaMaxTmp)        {thetaMax=thetaMaxTmp;}
   inline void setDeltaThetaMax(float thetaMaxTmp)        {deltaThetaMax=thetaMaxTmp;}
   inline void setRotate(float rotateTmp)        {rotate=rotateTmp;}
   inline void setNoPre(int noPreTmp)        {noPre=noPreTmp;}
   inline void setNoPost(int noPostTmp)        {noPost=noPostTmp;}

public:
//   int nxPatch_tmp;
//   int nyPatch_tmp;
//   int nfPatch_tmp;
//   int sx_tmp;
//   int sy_tmp;
//   int sf_tmp;
   float dxPost;
   float dyPost;
   float xDistHeadPreUnits;
   float yDistHeadPreUnits;
   int noPost;
   float dthPost;
   float th0Post;
   int noPre;
   int fPre;
   float thPre;
   float thetaMax;  // max orientation in units of PI
   float rotate;   // rotate so that axis isn't aligned
   float deltaThetaMax;  // max orientation in units of PI
   float deltaTheta;

   float calcDelta(int post, float dPost, float distHeadPreUnits);
};

} /* namespace PV */
#endif /* INITWEIGHTSPARAMS_HPP_ */
