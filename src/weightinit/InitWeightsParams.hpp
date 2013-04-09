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

   virtual void calcOtherParams(int patchIndex);
   float calcYDelta(int jPost);
   float calcXDelta(int iPost);

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

protected:
   int initialize_base();
   int initialize(HyPerConn * pConn);


   char * name; //this is actually the Connection name
   HyPerLayer     * pre;
   HyPerLayer     * post;
   HyPerCol       * parent;
   HyPerConn      * parentConn;
   ChannelType channel;    // which channel of the post to update (e.g. inhibit)

   void getcheckdimensionsandstrides();
   int kernelIndexCalculations(int patchIndex);

   //more get/set
   inline float getxDistHeadPreUnits()        {return xDistHeadPreUnits;}
   inline float getyDistHeadPreUnits()        {return yDistHeadPreUnits;}
   inline float getdyPost()        {return dyPost;}
   inline float getdxPost()        {return dxPost;}

public:
   float dxPost;
   float dyPost;
   float xDistHeadPreUnits;
   float yDistHeadPreUnits;


   float calcDelta(int post, float dPost, float distHeadPreUnits);
};

} /* namespace PV */
#endif /* INITWEIGHTSPARAMS_HPP_ */
