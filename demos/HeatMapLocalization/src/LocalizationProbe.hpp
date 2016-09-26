/*
 * LocalizationProbe.hpp
 *
 *  Created on: Sep 23, 2015
 *      Author: pschultz
 */

#ifndef LOCALIZATIONPROBE_HPP_
#define LOCALIZATIONPROBE_HPP_

#include <unistd.h>
#include <limits>
#include "probes/LayerProbe.hpp"
#include "layers/ImageFromMemoryBuffer.hpp"
#include "layers/HyPerLayer.hpp"
#include "LocalizationData.hpp"

/**
 * A probe to generate heat map montages.
 */
class LocalizationProbe: public PV::LayerProbe {
public:
   LocalizationProbe(const char * probeName, PV::HyPerCol * hc);
   virtual ~LocalizationProbe();

   char const * getImageLayerName() { return imageLayerName; }
   char const * getReconLayerName() { return reconLayerName; }
   char const * getClassNamesFile() { return classNamesFile; }

   /**
    * The root process returns the name of the class corresponding to feature k (zero-indexed).
    * If k is out of bounds or a nonroot process calls this method, NULL is returned.
    */
   char const * getClassName(int k);
   char const * getHeatMapMontageDir() { return heatMapMontageDir; }
   bool getDrawMontage() { return drawMontage; }
   char const * getDisplayCommand() { return displayCommand; }
   double getImageDilationX() { return imageDilationX; }
   double getImageDilationY() { return imageDilationY; }
   inline int getNumDisplayedCategories() const { return numDisplayedCategories; }
   inline float getDetectionThreshold(int idx) const { return (idx>=0 && idx<numDisplayedCategories) ? detectionThreshold[idx] : std::numeric_limits<float>::signaling_NaN(); }
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
   virtual int outputStateWrapper(double timef, double dt);

   /**
    * Returns the bounding box, feature, and score information for the detection of the given index.
    * Throws an out_of_range exception if index is >= getNumDetections().
    * The bounding box values (left, right, top, bottom) are in imageLayer coordinates.
    */
   inline LocalizationData const * getDetection(size_t index) { return &detections.at(index); }

   /**
    * Returns the vector of all detections that would be returned by getDetection(k)
    */
   inline std::vector<LocalizationData> const * getDetections() { return &detections; }

   /**
    * Sets the base of the output filename.  It takes everything after the
    * last slash '/' and before the last period '.' from the input argument
    * and stores it into the outputFilenameBase data member.
    * For example, if the input argument is
    * /home/user/Pictures/sample.images/image1.jpg
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
   LocalizationProbe();
   int initialize(const char * probeName, PV::HyPerCol * hc);
   virtual int ioParamsFillGroup(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_imageLayer(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_reconLayer(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_displayedCategories(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_displayCategoryIndexStart(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_displayCategoryIndexEnd(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_maxDetections(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_detectionThreshold(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_classNamesFile(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_outputPeriod(enum PV::ParamsIOFlag ioFlag);

   /**
    * The minimum width, in targetLayer pixels, for a bounding box to be
    * included in the detections.
    */
   virtual void ioParam_minBoundingBoxWidth(enum PV::ParamsIOFlag ioFlag);

   /**
    * The minimum height, in targetLayer pixels, for a bounding box to be
    * included in the detections.
    */
   virtual void ioParam_minBoundingBoxHeight(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_drawMontage(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_heatMapMaximum(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_heatMapMontageDir(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_imageBlendCoeff(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_boundingBoxLineWidth(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_displayCommand(enum PV::ParamsIOFlag ioFlag);
   virtual int initNumValues();
   virtual bool needUpdate(double timed, double dt);
   virtual int calcValues(double timevalue);
   virtual int outputState(double timevalue);

   int makeMontage();

private:
   int initialize_base();
   int setOptimalMontage();
   int findMaxLocation(int * winningFeature, int * winningIndex, int * xLocation, int * yLocation, float * maxActivity, float * buffer, PVLayerLoc const * loc);
   int findBoundingBox(int winningFeature, int winningIndex, int xLocation, int yLocation, float const * buffer, PVLayerLoc const * loc, int * boundingBox);
   int drawTextOnMontage(char const * backgroundColor, char const * textColor, char const * labelText, int xOffset, int yOffset, int width, int height);
   int drawTextIntoFile(char const * labelName, char const * backgroundColor, char const * textColor, char const * labelText, int width, int height=32);
   int insertFileIntoMontage(char const * labelname, int xOffset, int yOffset, int xExpectedSize, int yExpectedSize);
   int insertImageIntoMontage(int xStart, int yStart, pvadata_t const * sourceData, PVLayerLoc const * loc, bool extended);

   int makeGrayScaleImage();
   int drawHeatMaps();
   int drawOriginalAndReconstructed();
   int drawProgressInformation();
   int writeMontage();

// Member variables
protected:
   char * imageLayerName;
   char * reconLayerName;
   int minBoundingBoxWidth;
   int minBoundingBoxHeight;
   bool drawMontage;
   char * classNamesFile;
   char ** classNames; // The array of strings giving the names of each category.  Only the root process creates or uses this array.
   int * displayedCategories;
   int numDisplayedCategories;
   int displayCategoryIndexStart;
   int displayCategoryIndexEnd;
   char * heatMapMontageDir;
   std::vector<LocalizationData> detections;
   unsigned int maxDetections;
   int boundingBoxLineWidth;
   char * displayCommand;

   PV::HyPerLayer * imageLayer;
   PV::HyPerLayer * reconLayer;

   std::stringstream imagePVPFilePath;
   std::stringstream resultPVPFilePath;
   std::stringstream  reconPVPFilePath;

   double outputPeriod;
   double nextOutputTime; // Warning: this does not get checkpointed but it should.  Probes have no checkpointing infrastructure yet.

   char * outputFilenameBase;
   int numDetectionThresholds;
   float * detectionThreshold;
   int numHeatMapMaxima;
   float * heatMapMaximum;
   double imageDilationX; // The factor to multiply by to convert from targetLayer coordinates to imageLayer coordinates
   double imageDilationY; // The factor to multiply by to convert from targetLayer coordinates to imageLayer coordinates
   int numMontageRows;
   int numMontageColumns;
   int montageDimX;
   int montageDimY;
   pvadata_t * grayScaleImage;
   unsigned char * montageImage;
   unsigned char * montageImageLocal;
   unsigned char * montageImageComm;
   pvadata_t imageBlendCoeff; // heatmap image will be imageBlendCoeff * imagedata plus (1-imageBlendCoeff) * heatmap data
   int featurefieldwidth; // how many digits it takes to print the features (e.g. if nf was 100, the last feature is 99, which needs 2 digits)  Set in communicateInitInfo.  All processes compute this, although only the root process uses it
}; /* class LocalizationProbe */

PV::BaseObject * createLocalizationProbe(char const * name, PV::HyPerCol * hc);

#endif /* LOCALIZATIONPROBE_HPP_ */
