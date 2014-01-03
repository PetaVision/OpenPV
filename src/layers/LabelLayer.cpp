/*
 * LabelLayer.cpp
 *
 * Label Layer is a type of HyPerLayer that can be used to assign images with a label or category.
 * It is meant to be used together with a Movie layer.
 *
 * This layer takes a label from the filepath of the names of the images.  It has a layer with nf =
 * number of categories.  
 * [deprecated] For every new image, the layer will adjust its activity such that all
 * neurons have an activity of 0 except those with the feature corresponding to your image, which
 * will have an activity of 1.
 * LabelLayer is now normalized to unit L2 norm at each x,y location.
 * 
 *
 * Additional Params:
 * movieLayerName - the name of the movie layer, where images are defined in categories
 * labelStart - the character in the file name or file path (as defined by your [images].txt file)
 *      where the category or label is specified.  These labels must be formatted as a number
 *      between 0 and (total number of categories - 1), and every label must have the same number
 *      of digits (e.g. use 01 for 2 digits)
 * labelLength - the number of digits in the labels
 *
 * Other notes:
 * [deprecated] 
 * The label layer will automatically determine the nxscale and nyscale in order to have the smallest
 * number of neurons per layer possible.  You can specify an nxscale or nyscale, but these values will
 * be ignored.
 * The user can now specify a label layer of arbitray nx and ny
 *
 *  Created on: Jul 9, 2013
 *      Author: bcrocker
 *      modified: garkenyon
 */

#include "LabelLayer.hpp"

#ifdef PV_USE_MPI
   #include <mpi.h>
#endif
#include <string.h>
#include <sstream>

namespace PV{

LabelLayer::LabelLayer(){
   initialize_base();
}

LabelLayer::LabelLayer(const char * name, HyPerCol * hc, const char * movieLayerName){
   initialize_base();
   initialize(name, hc, movieLayerName);
}

LabelLayer::~LabelLayer(){
   free(movieLayerName);
}

int LabelLayer::initialize_base(){
   movieLayerName = NULL;
   movie = NULL;
   labelData = NULL;
   stepSize = 0;
   filename = NULL;
   currentLabel = -1;
   maxLabel = 0;
   beginLabel = -1;
   lenLabel = 0;
   echoLabelFlag = true;
   return PV_SUCCESS;
}

int LabelLayer::initialize(const char * name, HyPerCol * hc, const char * movieLayerName) {

   HyPerLayer::initialize(name, hc, 0);

   if (movieLayerName==NULL) {
      fprintf(stderr, "LabelLayer \"%s\" error: movieLayerName must be set.\n", name);
      abort();
   }
   this->movieLayerName = strdup(movieLayerName);
   if (this->movieLayerName==NULL) {
      fprintf(stderr, "LabelLayer \"%s\" error: unable to copy movieLayerName: %s\n", name, strerror(errno));
      abort();
   }

   this->labelLoc = * getLayerLoc();

   int status = PV_SUCCESS;

   return status;

}

int LabelLayer::communicateInitInfo() {
   int status = HyPerLayer::communicateInitInfo();

   HyPerLayer * hyperlayer = parent->getLayerFromName(movieLayerName);
   if (hyperlayer == NULL) {
      fprintf(stderr, "LabelLayer \"%s\" error: movieLayerName \"%s\" is not a layer in the HyPerCol.\n", name, movieLayerName);
      abort();
   }

   movie = dynamic_cast<Movie *>(hyperlayer);
   if (movie == NULL) {
      fprintf(stderr, "LabelLayer \"%s\" error: movieLayerName \"%s\" is not a Movie or Movie-derived class.\n", name, movieLayerName);
      abort();
   }

   return status;
}

int LabelLayer::allocateDataStructures() {
   int status = HyPerLayer::allocateDataStructures();

   free(clayer->V);
   clayer->V = NULL;

   PVParams * params = parent->parameters();

   this->beginLabel = params->value(name, "labelStart", beginLabel);
   this->maxLabel = params->value(name,"nf",maxLabel);
   this->lenLabel = params->value(name,"labelLength",lenLabel);
   this->echoLabelFlag = params->value(name,"echoLabelFlag",echoLabelFlag);

   labelData = clayer->activity->data;

   filename = movie->getCurrentImage();
   char tmp[lenLabel];
   for (int i=0; i<lenLabel; i++){
      tmp[i] = filename[i + beginLabel];
   }

   using std::istringstream;
   if ( ! (istringstream(tmp) >> currentLabel) ) currentLabel = -1;

   if (currentLabel == -1){
      status = PV_FAILURE;
      fprintf(stderr,"Current Label Integer: %d out of %d\n",currentLabel, maxLabel);
  }
   else{
      if (echoLabelFlag){
         fprintf(stderr,"Current Label Integer: %d out of %d\n",currentLabel, maxLabel);
      }

      // the firstlines below force an L2 norm of unity on the activity of the LabelLayer
      // the second lines force a stddev of 1
      for (int i = 0; i<(labelLoc.nf*(labelLoc.nx+labelLoc.nb*2)*(labelLoc.ny+labelLoc.nb*2)); i++){
         if (i%maxLabel == currentLabel){
            labelData[i] = sqrt(maxLabel-1)/sqrt(maxLabel);
            //labelData[i] = sqrt(maxLabel-1);
         }
         else{
            labelData[i] = -1/sqrt((maxLabel-1)*maxLabel);
            //labelData[i] = -1/sqrt((maxLabel-1));
         }
      }
   }

   return status;
}

int LabelLayer::updateState(double time, double dt){
   update_timer->start();

   int status = PV_SUCCESS;
   bool update = (movie->getLastUpdateTime()>lastUpdateTime);

   if (update){

      filename = movie->getCurrentImage();
      char tmp[lenLabel];
      for (int i=0; i<lenLabel; i++){
         tmp[i] = filename[i + beginLabel];
      }
      using std::istringstream;
      if ( ! (istringstream(tmp) >> currentLabel) ) currentLabel = -1;

      if (currentLabel == -1){
         status = PV_FAILURE;
         fprintf(stderr, "LabelLayer::updateState: currentLabel = %d", currentLabel);
         exit(PV_FAILURE);
      }
      else{

         if(echoLabelFlag){
            fprintf(stderr,"Current Label Integer: %d out of %d\n",currentLabel, maxLabel);
         }
         for (int i = 0; i<(labelLoc.nf*(labelLoc.nx+labelLoc.nb*2)*(labelLoc.ny+labelLoc.nb*2)); i++){
            if (i%maxLabel == currentLabel){
               labelData[i] = sqrt(maxLabel-1)/sqrt(maxLabel);
            }
            else{
               labelData[i] = -1/sqrt((maxLabel-1)*maxLabel);
            }
         }
      }
      lastUpdateTime = parent->simulationTime();

   }



   update_timer->stop();

   return status;

}

int LabelLayer::outputState(double time, bool last){
   return HyPerLayer::outputState(time, last);
}


//void LabelLayer::readNxScale(PVParams * params) {
//   int minX = this->parent->getNxGlobal();
//   nxScale = 1.0;
//   while (minX%2 == 0){
//      minX /= 2;
//      nxScale /=2;
//   }
//}

//void LabelLayer::readNyScale(PVParams * params) {
//   int minY = this->parent->getNyGlobal();
//   nyScale = 1.0;
//   while (minY%2 == 0){
//      minY /= 2;
//      nyScale /=2;
//   }
//}




// This layer exists to force LabelLayer to always have the smallest nx and ny
// dimension possible.
/*  // moved functionality to readNxScale and readNyScale
int LabelLayer::initClayer() {

   int minX = this->parent->getNxGlobal();
   int minY = this->parent->getNyGlobal();

   nxScale = 1.0;
   nyScale = 1.0;

   while (minX%2 == 0){
      minX /= 2;
      nxScale /=2;
   }
   while(minY%2 == 0){
      minY /= 2;
      nyScale /=2;
   }

   HyPerLayer::initClayer();

   return PV_SUCCESS;
}
*/


}


