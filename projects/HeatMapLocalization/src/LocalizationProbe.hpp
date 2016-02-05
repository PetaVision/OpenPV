/*
 * LocalizationProbe.hpp
 *
 *  Created on: Sep 23, 2015
 *      Author: pschultz
 */

#ifndef LOCALIZATIONPROBE_HPP_
#define LOCALIZATIONPROBE_HPP_

#include <unistd.h>
#include <io/LayerProbe.hpp>
#include <layers/ImageFromMemoryBuffer.hpp>
#include <layers/HyPerLayer.hpp>

/**
 * A probe to generate heat map montages.
 */
class LocalizationProbe: public PV::LayerProbe {
public:
   LocalizationProbe(const char * probeName, PV::HyPerCol * hc);
   virtual ~LocalizationProbe();

   char const * getImageLayerName() { return imageLayerName; }
   char const * getReconLayerName() { return reconLayerName; }
   char const * getOctaveCommand() { return octaveCommand; }
   char const * getOctaveLogFile() { return octaveLogFile; }
   char const * getClassNamesFile() { return classNamesFile; }
   char const * getClassName(int k);
   char const * getHeatMapMontageDir() { return heatMapMontageDir; }
   bool getDrawMontage() { return drawMontage; }
   char const * getDisplayCommand() { return displayCommand; }
   double getImageDilationX() { return imageDilationX; }
   double getImageDilationY() { return imageDilationY; }
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();


// // Not used by harness since we don't have a filename to use for the base
// /**
//  * sets the base of the output filename.  It takes the part of the string
//  * Everything before the last slash '/' is removed, and then everything
//  * from the last period '.' onward is removed.  For example, if the input
//  * argument is /home/user/Pictures/sample.images/image1.jpg
//  * then the output filename base will be set to image1
//  * If there is no slash in the input string, nothing is removed from the
//  * front of the string; if there is no period (except before a slash)
//  * nothing is discarded from the end.
//  * Note that the input argument itself is not modified, only the
//  * outputFilenameBase member variable.
//  */
// int setOutputFilenameBase(char const * fn);
   char const * getOutputFilenameBase() { return outputFilenameBase; }

protected:
   LocalizationProbe();
   int initialize(const char * probeName, PV::HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_imageLayer(enum ParamsIOFlag ioFlag);
   virtual void ioParam_reconLayer(enum ParamsIOFlag ioFlag);
   virtual void ioParam_detectionThreshold(enum ParamsIOFlag ioFlag);
   virtual void ioParam_heatMapMaximum(enum ParamsIOFlag ioFlag);
   virtual void ioParam_classNamesFile(enum ParamsIOFlag ioFlag);
   virtual void ioParam_outputPeriod(enum ParamsIOFlag ioFlag);
   virtual void ioParam_drawMontage(enum ParamsIOFlag ioFlag);
   virtual void ioParam_octaveCommand(enum ParamsIOFlag ioFlag);
   virtual void ioParam_octaveLogFile(enum ParamsIOFlag ioFlag);
   virtual void ioParam_displayCategoryIndexStart(enum ParamsIOFlag ioFlag);
   virtual void ioParam_displayCategoryIndexEnd(enum ParamsIOFlag ioFlag);
   virtual void ioParam_heatMapMontageDir(enum ParamsIOFlag ioFlag);
   virtual void ioParam_imageBlendCoeff(enum ParamsIOFlag ioFlag);
   virtual void ioParam_displayCommand(enum ParamsIOFlag ioFlag);
   virtual int initNumValues();
   int makeDisplayCategoryIndicesString();
   virtual bool needUpdate(double timed, double dt);
   virtual int calcValues(double timevalue); // Currently, those values are computed by octave, so not convenient to retrieve.
   virtual int outputState(double timevalue);

   int makeMontage();

private:
   int initialize_base();
   int setOptimalMontage();
   int findMaxLocation(int * winningFeature, int * xLocation, int * yLocation, pvadata_t * maxActivity);
   int findBoundingBox(int winningFeature, int xLocation, int yLocation, int * boundingBox);
   int makeMontageLabelfile(char const * labelName, char const * backgroundColor, char const * textColor, char const * labelText);
   int insertLabelIntoMontage(char const * labelname, int xOffset, int yOffset, int xExpectedSize, int yExpectedSize);

// Member variables
protected:
   char * imageLayerName;
   char * reconLayerName;
   bool drawMontage;
   char * octaveCommand;
   char * octaveLogFile;
   char * classNamesFile;
   char ** classNames;
   int displayCategoryIndexStart;
   int displayCategoryIndexEnd;
   char * heatMapMontageDir;
   char * displayCommand;

   PV::HyPerLayer * imageLayer;
   PV::HyPerLayer * reconLayer;

   std::stringstream imagePVPFilePath;
   std::stringstream resultPVPFilePath;
   std::stringstream  reconPVPFilePath;

   double outputPeriod;
   double nextOutputTime; // Warning: this does not get checkpointed but it should.  Probes have no checkpointing infrastructure yet.
   pid_t octavePid;

   char * outputFilenameBase;
   float detectionThreshold; // Should become an array
   float heatMapMaximum;
   double imageDilationX; // The factor to multiply by to convert from targetLayer coordinates to imageLayer coordinates
   double imageDilationY; // The factor to multiply by to convert from targetLayer coordinates to imageLayer coordinates
   int numMontageRows;
   int numMontageColumns;
   int montageDimX;
   int montageDimY;
   pvadata_t * grayScaleImage;
   unsigned char * montageImage;
   pvadata_t imageBlendCoeff; // heatmap image will be imageBlendCoeff * imagedata plus (1-imageBlendCoeff) * heatmap data
   int featurefieldwidth; // how many digits it takes to print the features (e.g. if nf was 100, the last feature is 99, which needs 2 digits)  Set in communicateInitInfo.
}; /* class LocalizationProbe */

#endif /* LOCALIZATIONPROBE_HPP_ */
