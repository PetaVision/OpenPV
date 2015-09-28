/*
 * HeatMapProbe.h
 *
 *  Created on: Sep 23, 2015
 *      Author: pschultz
 */

#ifndef HEATMAPPROBE_HPP_
#define HEATMAPPROBE_HPP_

#include <unistd.h>
#include <io/ColProbe.hpp>
#include <layers/ImageFromMemoryBuffer.hpp>
#include <layers/HyPerLayer.hpp>

/**
 * A probe to generate heat map montages.
 */
class HeatMapProbe: public PV::ColProbe {
public:
   HeatMapProbe(const char * probeName, PV::HyPerCol * hc);
   virtual ~HeatMapProbe();

   char const * getImageLayerName() { return imageLayerName; }
   char const * getResultLayerName() { return resultLayerName; }
   char const * getReconLayerName() { return reconLayerName; }
   char const * getResultTextFile() { return resultTextFile; }
   char const * getOctaveCommand() { return octaveCommand; }
   char const * getOctaveLogFile() { return octaveLogFile; }
   char const * getClassNames() { return classNames; }
   char const * getEvalCategoryIndices() { return evalCategoryIndices; }
   char const * getDisplayCategoryIndices() { return displayCategoryIndices; }
   char const * getHighlightThreshold() { return highlightThreshold; }
   char const * getHeatMapThreshold() { return heatMapThreshold; }
   char const * getHeatMapMaximum() { return heatMapMaximum; }
   char const * getDrawBoundingBoxes() { return drawBoundingBoxes; }
   char const * getBoundingBoxThickness() { return boundingBoxThickness; }
   char const * getDbscanEps() { return dbscanEps; }
   char const * getDbscanDensity() { return dbscanDensity; }
   char const * getHeatMapMontageDir() { return heatMapMontageDir; }
   char const * getDisplayCommand() { return displayCommand; }

protected:
   HeatMapProbe();
   int initialize(const char * probeName, PV::HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_reconLayer(enum ParamsIOFlag ioFlag);
   virtual void ioParam_outputPeriod(enum ParamsIOFlag ioFlag);
   virtual int communicateInitInfo();
   virtual int initNumValues();
   virtual bool needUpdate(double timed, double dt);
   virtual bool needRecalc(double timevalue) { return false; } // Perhaps TODO: values are the confidences of each category.
   virtual int calcValues(double timevalue) { return 0; } // Currently, those values are computed by octave, so not convenient to retrieve.
   virtual int outputState(double timevalue);
   int waitOctaveFinished();
   int octaveProcess();

   // writeBufferFile and gatherActivity hackily copy code from HyPerLayer and gatherActivity until I debug the float const specialization of the templates.
   static int writeBufferFile(char const * filename, PV::InterColComm * comm, double timevalue, pvadata_t const * A, PVLayerLoc const * loc);
   static int gatherActivity(PV_Stream * pvstream, PV::Communicator * comm, int rootproc, pvadata_t const * buffer, const PVLayerLoc * layerLoc);

private:
   int initialize_base();

// Member variables
protected:
   // config file parameters
   char * imageLayerName;
   char * resultLayerName;
   char * reconLayerName;
   char * resultTextFile;
   char * octaveCommand;
   char * octaveLogFile;
   char * classNames;
   char * evalCategoryIndices;
   char * displayCategoryIndices;
   char * highlightThreshold;
   char * heatMapThreshold;
   char * heatMapMaximum;
   char * drawBoundingBoxes;
   char * boundingBoxThickness;
   char * dbscanEps;
   char * dbscanDensity;
   char * heatMapMontageDir;
   char * displayCommand;

   PV::ImageFromMemoryBuffer * imageLayer;
   PV::HyPerLayer * resultLayer;
   PV::HyPerLayer * reconLayer;

   std::stringstream imagePVPFilePath;
   std::stringstream resultPVPFilePath;
   std::stringstream  reconPVPFilePath;

   double outputPeriod;
   double nextOutputTime; // Warning: this does not get checkpointed but it should.  Probes have no checkpointing infrastructure yet.
   pid_t octavePid;
}; /* class HeatMapProbe */

#endif /* HEATMAPPROBE_HPP_ */
