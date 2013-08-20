/*
 * LabelLayer.cpp
 *
 *  Created on: Jul 9, 2013
 *      Author: bcrocker
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

   // Moved to communicateInitInfo()
   // HyPerLayer * hyperlayer = parent->getLayerFromName(movieLayerName);
   // if (hyperlayer == NULL) {
   //    fprintf(stderr, "LabelLayer \"%s\" error: movieLayerName \"%s\" is not a layer in the HyPerCol.\n", name, movieLayerName);
   //    abort();
   // }
   //
   // movie = dynamic_cast<Movie *>(hyperlayer);
   // if (movie == NULL) {
   //    fprintf(stderr, "LabelLayer \"%s\" error: movieLayerName \"%s\" is not a Movie or Movie-derived class.\n", name, movieLayerName);
   //    abort();
   // }

   this->labelLoc = * getLayerLoc();

//   int minX = labelLoc.nx;
//   int minY = labelLoc.ny;
//
//   while (minX%2 == 0){
//      minX = minX/2;
//   }
//   while(minY%2 == 0){
//      minY = minY/2;
//   }
//
//   labelLoc.nx = minX;
//   labelLoc.ny = minY;

   int status = PV_SUCCESS;

//   // Moved to allocateDataStructures()
//   free(clayer->V);
//   clayer->V = NULL;
//
//   PVParams * params = hc->parameters();
//
//   this->beginLabel = params->value(name, "labelStart", beginLabel);
//   this->maxLabel = params->value(name,"nf",maxLabel);
//   this->lenLabel = params->value(name,"labelLength",lenLabel);
//
//   labelData = clayer->activity->data;
//
//   filename = movie->getCurrentImage();
//   char tmp[lenLabel];
//   for (int i=0; i<lenLabel; i++){
//      tmp[i] = filename[i + beginLabel];
//   }
//
//   using std::istringstream;
//   if ( ! (istringstream(tmp) >> currentLabel) ) currentLabel = -1;
//
//   if (currentLabel == -1){
//      status = PV_FAILURE;
//   }
//   else{
//
//      fprintf(stderr,"Current Label Integer: %d out of %d\n",currentLabel, maxLabel);
//
//      //fprintf(stderr,"NF = %d, NX = %d, NY = %d",labelLoc.nf, labelLoc.nx, labelLoc.ny);
//      for (int i = 0; i<(labelLoc.nf*(labelLoc.nx+labelLoc.nb*2)*(labelLoc.ny+labelLoc.nb*2)); i++){
//         if (i%maxLabel == currentLabel){
//            labelData[i] = 1.0;
//         }
//         else{
//            labelData[i] = 0.0;
//         }
//      }
//   }

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
   }
   else{

      fprintf(stderr,"Current Label Integer: %d out of %d\n",currentLabel, maxLabel);

      fprintf(stderr,"NF = %d, NX = %d, NY = %d",labelLoc.nf, labelLoc.nx, labelLoc.ny);
      for (int i = 0; i<(labelLoc.nf*(labelLoc.nx+labelLoc.nb*2)*(labelLoc.ny+labelLoc.nb*2)); i++){
         if (i%maxLabel == currentLabel){
            labelData[i] = 1.0;
         }
         else{
            labelData[i] = 0.0;
         }
      }
   }

   return status;
}

int LabelLayer::updateState(double time, double dt){
   update_timer->start();

   int status = PV_SUCCESS;
   bool update = movie->getNewImageFlag();

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
      }
      else{

         fprintf(stderr,"Current Label Integer: %d out of %d\n",currentLabel, maxLabel);

         //fprintf(stderr,"NF = %d, NX = %d, NY = %d",labelLoc.nf, labelLoc.nx, labelLoc.ny);
         for (int i = 0; i<(labelLoc.nf*(labelLoc.nx+labelLoc.nb*2)*(labelLoc.ny+labelLoc.nb*2)); i++){
            //fprintf(stderr,"i = %d, i mod maxlabel = %d\n",i,i%maxLabel);
            if (i%maxLabel == currentLabel){
               labelData[i] = 1.0;
            }
            else{
               labelData[i] = 0.0;
            }
         }
      }

   }



   update_timer->stop();

   return status;

}

int LabelLayer::outputState(double time, bool last){
   // io_timer->start();
   // fprintf(stderr,"Writing Label Layer state \n");
   // io_timer->stop();

   // HyPerLayer::outputState already has an io timer so don't duplicate
   return HyPerLayer::outputState(time, last);
}


// This layer exists to force LabelLayer to always have the smallest nx and ny
// dimension possible.
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


}


