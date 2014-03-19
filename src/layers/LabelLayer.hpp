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

namespace PV{

class LabelLayer : public HyPerLayer {

protected:
   LabelLayer();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_movieLayerName(enum ParamsIOFlag ioFlag);
   virtual void ioParam_labelStart(enum ParamsIOFlag ioFlag);
   virtual void ioParam_labelLength(enum ParamsIOFlag ioFlag);
   virtual void ioParam_echoLabelFlag(enum ParamsIOFlag ioFlag);
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
   virtual int updateState(double time, double dt);
   virtual int outputState(double time, bool last);
   virtual ~LabelLayer();
#ifdef OBSOLETE
   //Obsolete Jan 15th, 2014 by slundquist
   //getLastUpdateTime in HyPerLayer no longer updates lastUpdateTime, so no longer need to override
   virtual double getLastUpdateTime() { return lastUpdateTime;}
#endif // OBSOLETE

private:
   int initialize_base();

};

}


#endif /* LABELLAYER_HPP_ */
