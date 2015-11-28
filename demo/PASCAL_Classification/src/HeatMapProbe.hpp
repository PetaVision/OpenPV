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

   /**
    * sets the base of the output filename.  It takes the part of the string
    * Everything before the last slash '/' is removed, and then everything
    * from the last period '.' onward is removed.  For example, if the input
    * argument is /home/user/Pictures/sample.images/image1.jpg
    * then the output filename base will be set to image1
    * If there is no slash in the input string, nothing is removed from the
    * front of the string; if there is no period (except before a slash)
    * nothing is discarded from the end.
    * Note that the input argument itself is not modified, only the
    * outputFilenameBase member variable.
    */
   int setOutputFilenameBase(char const * fn);
   char const * getOutputFilenameBase() { return outputFilenameBase; }

protected:
   HeatMapProbe();
   int initialize(const char * probeName, PV::HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_confidenceTable(enum ParamsIOFlag ioFlag);
   virtual void ioParam_imageLayer(enum ParamsIOFlag ioFlag);
   virtual void ioParam_resultLayer(enum ParamsIOFlag ioFlag);
   virtual void ioParam_reconLayer(enum ParamsIOFlag ioFlag);
   virtual void ioParam_classNames(enum ParamsIOFlag ioFlag);
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
   static int writeBufferFile(char const * filename, PV::InterColComm * comm, double timevalue, pvadata_t * A, PVLayerLoc const * loc);
   static int gatherActivity(PV_Stream * pvstream, PV::Communicator * comm, int rootproc, pvadata_t * buffer, const PVLayerLoc * layerLoc);

private:
   int initialize_base();

// Member variables
protected:
   // config file parameters
   char * confidenceTable;
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

   char * outputFilenameBase;
}; /* class HeatMapProbe */

#endif /* HEATMAPPROBE_HPP_ */
