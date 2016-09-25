/*
 * LocalizationBBFindProbe.hpp
 *
 *  Created on: Sep 23, 2015
 *      Author: pschultz
 */

#ifndef LOCALIZATIONBBFINDPROBE_HPP_
#define LOCALIZATIONBBFINDPROBE_HPP_

#include <unistd.h>
#include <limits>
#include "probes/LayerProbe.hpp"
#include "layers/ImageFromMemoryBuffer.hpp"
#include "layers/HyPerLayer.hpp"
#include "BBFind.hpp"
#include "LocalizationProbe.hpp"

/**
 * A probe to generate heat map montages using BBFind.
 */
class LocalizationBBFindProbe: public LocalizationProbe {
public:
   LocalizationBBFindProbe(const char * probeName, PV::HyPerCol * hc);
   virtual ~LocalizationBBFindProbe();

   int communicateInitInfo();

protected:
   LocalizationBBFindProbe();
   int initialize(const char * probeName, PV::HyPerCol * hc);
   virtual int ioParamsFillGroup(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_framesPerMap(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_threshold(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_contrast(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_contrastStrength(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_prevInfluence(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_accumulateAmount(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_prevLeakTau(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_minBlobSize(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_boundingboxGuessSize(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_slidingAverageSize(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_maxRectangleMemory(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_detectionWait(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_internalMapWidth(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_internalMapHeight(enum PV::ParamsIOFlag ioFlag);

   virtual int calcValues(double timevalue);
   double computeBoxConfidence(LocalizationData const& bbox, pvadata_t const * buffer, int nx, int ny, int nf);

private:
   int initialize_base();

// Member variables
protected:
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

   BBFind bbfinder;
}; /* class LocalizationBBFindProbe */

PV::BaseObject * createLocalizationBBFindProbe(char const * name, PV::HyPerCol * hc);

#endif /* LOCALIZATIONBBFINDPROBE_HPP_ */
