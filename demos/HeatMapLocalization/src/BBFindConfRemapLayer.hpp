/*
 * BBFindConfRemapLayer.hpp
 *
 *  Created on: May 17, 2016
 *      Author: pschultz
 */

#ifndef BBFINDCONFREMAPLAYER_HPP_
#define BBFINDCONFREMAPLAYER_HPP_

#include "layers/HyPerLayer.hpp"
#include "BBFind.hpp"

class BBFindConfRemapLayer: public PV::HyPerLayer {
public:

   BBFindConfRemapLayer(char const * name, PV::HyPerCol * hc);
   virtual ~BBFindConfRemapLayer();

   bool activityIsSpiking() { return false; }
   int communicateInitInfo();
   int allocateDataStructures();
   double getDeltaUpdateTime();
   virtual int updateState(double t, double dt);

   // public accessor methods
   int getNumDisplayedCategories() { return numDisplayedCategories; }
   int const * getDisplayedCategories() { return displayedCategories; }
   std::vector<BBFind> const getBoundingBoxFinder() { return boundingboxFinder; }

protected:
   BBFindConfRemapLayer();
   int initialize(char const * name, PV::HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_displayedCategories(enum ParamsIOFlag ioFlag);
   virtual void ioParam_imageLayer(enum ParamsIOFlag ioFlag);
   virtual void ioParam_framesPerMap(enum ParamsIOFlag ioFlag);
   virtual void ioParam_threshold(enum ParamsIOFlag ioFlag);
   virtual void ioParam_contrast(enum ParamsIOFlag ioFlag);
   virtual void ioParam_contrastStrength(enum ParamsIOFlag ioFlag);
   virtual void ioParam_prevInfluence(enum ParamsIOFlag ioFlag);
   virtual void ioParam_accumulateAmount(enum ParamsIOFlag ioFlag);
   virtual void ioParam_prevLeakTau(enum ParamsIOFlag ioFlag);
   virtual void ioParam_minBlobSize(enum ParamsIOFlag ioFlag);
   virtual void ioParam_boundingboxGuessSize(enum ParamsIOFlag ioFlag);
   virtual void ioParam_slidingAverageSize(enum ParamsIOFlag ioFlag);
   virtual void ioParam_maxRectangleMemory(enum ParamsIOFlag ioFlag);
   virtual void ioParam_detectionWait(enum ParamsIOFlag ioFlag);
   virtual void ioParam_internalMapWidth(enum ParamsIOFlag ioFlag);
   virtual void ioParam_internalMapHeight(enum ParamsIOFlag ioFlag);

private:
   int initialize_base();
   void setBoundingBoxFinderParams(BBFind& bbf);

// Data members
private:
   char * imageLayerName = NULL;
   HyPerLayer * imageLayer = NULL;
   int framesPerMap;
   float threshold;
   float contrast;
   float contrastStrength;
   float prevInfluence;
   float accumulateAmount;
   float prevLeakTau;
   int minBlobSize;
   int boundingboxGuessSize;
   int slidingAverageSize;
   int maxRectangleMemory;
   int detectionWait;
   int internalMapWidth;
   int internalMapHeight;

   int imageWidth;
   int imageHeight;

   std::vector<BBFind> boundingboxFinder;
   int * displayedCategories = NULL;
   int numDisplayedCategories = 0;
}; // class BBFindConfRemapLayer

PV::BaseObject * createBBFindConfRemapLayer(char const * name, PV::HyPerCol * hc);

#endif /* BBFINDCONFREMAPLAYER_HPP_ */
