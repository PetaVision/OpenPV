/*
 * LabelLayer.hpp
 *
 *  Created on: Jul 9, 2013
 *      Author: bcrocker
 */

#ifndef LABELLAYER_HPP_
#define LABELLAYER_HPP_

#include "HyPerLayer.hpp"
#include "../columns/HyPerCol.hpp"
#include "Movie.hpp"
#include <cMakeHeader.h>

namespace PV{

class LabelLayer : public HyPerLayer {

#ifdef PV_USE_GDAL

protected:
   LabelLayer();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_InitVType(enum ParamsIOFlag ioFlag);
   virtual void ioParam_movieLayerName(enum ParamsIOFlag ioFlag);
   virtual void ioParam_labelStart(enum ParamsIOFlag ioFlag);
   virtual void ioParam_labelLength(enum ParamsIOFlag ioFlag);
   virtual void ioParam_echoLabelFlag(enum ParamsIOFlag ioFlag);
   virtual int allocateV();
   virtual int initializeActivity();
   char * movieLayerName;
   Movie * movie;
   pvdata_t * labelData;
   int stepSize;
   const char * filename;
   int currentLabel;
   int maxLabel;
   int beginLabel;
   int lenLabel;
   bool echoLabelFlag;
   PVLayerLoc labelLoc;


public:
   LabelLayer(const char * name, HyPerCol * hc);
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
   virtual bool activityIsSpiking() { return false; }
   virtual int updateState(double time, double dt);
   virtual int outputState(double time, bool last);
   virtual ~LabelLayer();

private:
   int initialize_base();
   int updateLabels();
#else // PV_USE_GDAL
public:
   LabelLayer(char const * name, HyPerCol * hc);
protected:
   LabelLayer();
#endif // PV_USE_GDAL

}; // class LabelLayer

BaseObject * createLabelLayer(char const * name, HyPerCol * hc);

}  // namespace PV

#endif /* LABELLAYER_HPP_ */
