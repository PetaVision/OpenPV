/*
 * BBFindConfRemapProbe.hpp
 *
 *  Created on: May 18, 2016
 *      Author: pschultz
 */

#ifndef BBFINDCONFREMAPPROBE_HPP_
#define BBFINDCONFREMAPPROBE_HPP_

#include "io/LayerProbe.hpp"
#include "LocalizationData.hpp"
#include "BBFindConfRemapLayer.hpp"

class BBFindConfRemapProbe: public PV::LayerProbe {
public:
   BBFindConfRemapProbe(char const * name, PV::HyPerCol * hc);
   virtual ~BBFindConfRemapProbe();

   void setOutputFilenameBase(char const * fn);
   int communicateInitInfo();
   int allocateDataStructures();
   virtual int outputStateWrapper(double t, double dt);

protected:
   BBFindConfRemapProbe();
   int initialize(char const * name, PV::HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_imageLayer(enum ParamsIOFlag ioFlag);
   virtual void ioParam_reconLayer(enum ParamsIOFlag ioFlag);
   virtual void ioParam_classNamesFile(enum ParamsIOFlag ioFlag);
   virtual void ioParam_minBoundingBoxWidth(enum ParamsIOFlag ioFlag);
   virtual void ioParam_minBoundingBoxHeight(enum ParamsIOFlag ioFlag);
   virtual void ioParam_drawMontage(enum ParamsIOFlag ioFlag);
   virtual void ioParam_heatMapMontageDir(enum ParamsIOFlag ioFlag);
   virtual void ioParam_heatMapThreshold(enum ParamsIOFlag ioFlag);
   virtual void ioParam_heatMapMaximum(enum ParamsIOFlag ioFlag);
   virtual void ioParam_imageBlendCoeff(enum ParamsIOFlag ioFlag);
   virtual void ioParam_boundingBoxLineWidth(enum ParamsIOFlag ioFlag);
   virtual void ioParam_displayCommand(enum ParamsIOFlag ioFlag);
   virtual int calcValues(double timevalue);
   virtual int outputState(double timevalue);
   int makeMontage(int b);

private:
   int initialize_base();
   void setLayerFromParam(PV::HyPerLayer ** layer, char const * layerType, char const * layerName);
   void setOptimalMontage();
   void makeGrayScaleImage(int b);
   void drawHeatMaps(int b);
   void drawOriginalAndReconstructed();
   void drawProgressInformation();
   void drawTextOnMontage(char const * backgroundColor, char const * textColor, char const * labelText, int xOffset, int yOffset, int width, int height);
   void drawTextIntoFile(char const * labelName, char const * backgroundColor, char const * textColor, char const * labelText, int width, int height=32);
   void insertFileIntoMontage(char const * labelname, int xOffset, int yOffset, int xExpectedSize, int yExpectedSize);
   void insertImageIntoMontage(int xStart, int yStart, pvadata_t const * sourceData, PVLayerLoc const * loc, bool extended);
   void writeMontage();

protected:
   BBFindConfRemapLayer * targetBBFindConfRemapLayer = NULL;
   char * imageLayerName = NULL;
   PV::HyPerLayer * imageLayer = NULL;
   char * reconLayerName = NULL;
   PV::HyPerLayer * reconLayer = NULL;
   char * classNamesFile = NULL;
   char ** classNames = NULL; // The array of strings giving the names of each category.  Only the root process creates or uses this array.
   int minBoundingBoxWidth = 6;
   int minBoundingBoxHeight = 6;
   bool drawMontage = false;
   char * heatMapMontageDir = NULL;
   int numHeatMapThresholds = 0;
   float * heatMapThreshold = NULL;
   int numHeatMapMaxima = 0;
   float * heatMapMaximum = NULL;
   pvadata_t imageBlendCoeff; // heatmap image will be imageBlendCoeff * imagedata plus (1-imageBlendCoeff) * heatmap data
   int boundingBoxLineWidth = 5;
   char * displayCommand = NULL;
   vector<vector<LocalizationData>> detectionS;

   int numMontageRows = -1;
   int numMontageColumns = -1;
   int montageDimX = -1;
   int montageDimY = -1;
   unsigned char * montageImage = NULL;
   unsigned char * montageImageLocal = NULL;
   unsigned char * montageImageComm = NULL;
   pvadata_t * grayScaleImage = NULL;
   double imageDilationX = 1.0;
   double imageDilationY = 1.0;
   char * outputFilenameBase = NULL;
}; // class BBFindConfRemapProbe


PV::BaseObject * createBBFindConfRemapProbe(char const * name, PV::HyPerCol * hc);

#endif /* BBFINDCONFREMAPPROBE_HPP_ */
