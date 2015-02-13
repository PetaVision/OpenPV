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
   InitWeightsParams(char const * name, HyPerCol * hc);
   virtual ~InitWeightsParams();

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual int communicateParamsInfo();

   //get-set methods:
   inline const char * getName()                {return name;}
   inline void setName(const char * name)    {free(this->name); this->name = strdup(name);}
   inline HyPerCol * getParent()                {return parent;}
   inline HyPerLayer * getPre()                 {return pre;}
   inline HyPerLayer * getPost()                {return post;}
   inline HyPerConn * getParentConn()           {return parentConn;}
   inline ChannelType getChannel()              {return channel;}
   inline const char * getFilename()            {return filename;}
   inline bool getUseListOfArborFiles()         {return useListOfArborFiles;}
   inline bool getCombineWeightFiles()          {return combineWeightFiles;}
   inline int getNumWeightFiles()               {return numWeightFiles;}

   virtual void calcOtherParams(int patchIndex);
   float calcYDelta(int jPost);
   float calcXDelta(int iPost);
   float calcDelta(int post, float dPost, float distHeadPreUnits);

   //get/set:
   int getnfPatch();
   int getnyPatch();
   int getnxPatch();
   int getPatchSize();
   int getsx();
   int getsy();
   int getsf();

   float getWMin();     // minimum allowed weight value
   float getWMax();     // maximum allowed weight value

protected:
   int initialize_base();
   int initialize(char const * name, HyPerCol * hc);

   char * name; //this is actually the Connection name
   HyPerLayer     * pre;
   HyPerLayer     * post;
   HyPerCol       * parent;
   HyPerConn      * parentConn;
   ChannelType channel;    // which channel of the post to update (e.g. inhibit)
   char * filename;
   bool useListOfArborFiles;
   bool combineWeightFiles;
   int numWeightFiles;

   void getcheckdimensionsandstrides();
   int kernelIndexCalculations(int patchIndex);
   virtual void ioParam_initWeightsFile(enum ParamsIOFlag ioFlag);
   virtual void ioParam_useListOfArborFiles(enum ParamsIOFlag ioFlag);
   virtual void ioParam_combineWeightFiles(enum ParamsIOFlag ioFlag);
   virtual void ioParam_numWeightFiles(enum ParamsIOFlag ioFlag);
   //more get/set
   inline float getxDistHeadPreUnits()   {return xDistHeadPreUnits;}
   inline float getyDistHeadPreUnits()   {return yDistHeadPreUnits;}
   inline float getdyPost()              {return dyPost;}
   inline float getdxPost()              {return dxPost;}

   float dxPost;
   float dyPost;
   float xDistHeadPreUnits;
   float yDistHeadPreUnits;
};

} /* namespace PV */
#endif /* INITWEIGHTSPARAMS_HPP_ */
