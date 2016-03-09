/*
 * ConvertFromTable.cpp
 *
 *  Created on: Dec 1, 2015
 *      Author: pschultz
 */

#include "ConvertFromTable.hpp"
#define CONVTABLEHEADERSIZE (sizeof(char)*(size_t) 8+sizeof(int)+sizeof(int)+sizeof(float)+sizeof(float))

ConvertFromTable::ConvertFromTable(char const * name, PV::HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

ConvertFromTable::ConvertFromTable() {
   initialize_base();
}

int ConvertFromTable::initialize_base() {
   dataFile = NULL;
   memset(&convTable, 0, sizeof(convTableStruct));
   convData = NULL;
   return PV_SUCCESS;
}

int ConvertFromTable::initialize(char const * name, PV::HyPerCol * hc) {
   int status = PV::CloneVLayer::initialize(name, hc);
   return status;
}

int ConvertFromTable::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = PV::CloneVLayer::ioParamsFillGroup(ioFlag);
   ioParam_dataFile(ioFlag);
   return status;
}

void ConvertFromTable::ioParam_dataFile(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "dataFile", &dataFile);
}

int ConvertFromTable::allocateDataStructures() {
   int status = PV::CloneVLayer::allocateDataStructures();
   if (status==PV_SUCCESS) {
      status = loadConversionTable();
   }
   return status;
}

int ConvertFromTable::loadConversionTable() {
   int status = PV_SUCCESS;
   if (parent->globalRank()==0) {
      FILE * conversionTableFP = fopen(dataFile, "r");
      if (conversionTableFP==NULL) {
         fprintf(stderr, "%s \"%s\": error opening dataFile \"%s\": %s\n",
               getKeyword(), getName(), dataFile, strerror(errno));
         exit(EXIT_FAILURE);
      }
      status = fseek(conversionTableFP, 0L, SEEK_END);
      if (status != 0) {
         fprintf(stderr, "%s \"%s\": error seeking to end off dataFile \"%s\": %s\n",
               getKeyword(), getName(), dataFile, strerror(errno));
         exit(EXIT_FAILURE);
      }
      long fileLength = ftell(conversionTableFP);
      if (fileLength<0) {
         fprintf(stderr, "%s \"%s\": error getting length of dataFile \"%s\": %s\n",
               getKeyword(), getName(), dataFile, strerror(errno));
         exit(EXIT_FAILURE);
      }
      if (fileLength==0) {
         fprintf(stderr, "%s \"%s\": dataFile \"%s\" is empty.\n",
               getName(), getKeyword(), dataFile);
         exit(EXIT_FAILURE);
      }
      if (fileLength<CONVTABLEHEADERSIZE) {
         fprintf(stderr, "%s \"%s\": dataFile \"%s\" is too small (%ld bytes; minimum is %zu bytes)\n",
               getKeyword(), getName(), dataFile, fileLength, CONVTABLEHEADERSIZE);
         exit(EXIT_FAILURE);
      }
      rewind(conversionTableFP);
      size_t numRead = (size_t) 0;
      char hdr[8];
      numRead += fread(hdr, sizeof(char), 8, conversionTableFP);
      numRead += fread(&convTable.numPoints, 1, sizeof(int), conversionTableFP);
      numRead += fread(&convTable.numFeatures, 1, sizeof(int), conversionTableFP);
      numRead += fread(&convTable.minRecon, 1, sizeof(int), conversionTableFP);
      numRead += fread(&convTable.maxRecon, 1, sizeof(int), conversionTableFP);
      if(numRead != CONVTABLEHEADERSIZE) {
         fprintf(stderr, "%s \"%s\": error reading header of dataFile \"%s\": expected %zu bytes; only read %zu\n",
               getName(), getKeyword(), dataFile, CONVTABLEHEADERSIZE, numRead);
         exit(EXIT_FAILURE);
      }
      if (memcmp(hdr, "convTabl", 8)) {
         fprintf(stderr, "%s \"%s\": dataFile \"%s\" does not have the expected header.\n",
               getKeyword(), getName(), dataFile);
         exit(EXIT_FAILURE);
      }
      int correctFileSize = (int) (sizeof(float)*convTable.numPoints*convTable.numFeatures+CONVTABLEHEADERSIZE);
      if (correctFileSize != fileLength) {
         fprintf(stderr, "%s \"%s\": dataFile \"%s\" has unexpected length (%d-by-%d %zu-byte data values with %zu-byte header, but file size is %ld\n",
               getKeyword(), getName(), dataFile, convTable.numPoints, convTable.numFeatures, sizeof(float), CONVTABLEHEADERSIZE, fileLength);
         exit(EXIT_FAILURE);
      }
      if (convTable.numFeatures != getLayerLoc()->nf) {
         fprintf(stderr, "%s \"%s\" error: layer has %d features but dataFile \"%s\" has numFeatures=%d\n",
               getKeyword(), getName(), getLayerLoc()->nf, dataFile, convTable.numFeatures);
         exit(EXIT_FAILURE);
      }
      if (convTable.numFeatures < 2) {
         fprintf(stderr, "%s \"%s\" error: dataFile \"%s\" has numPoints=%d but must be >= 2.\n",
               getKeyword(), getName(), dataFile, convTable.numPoints);
         exit(EXIT_FAILURE);
      }
      size_t numValues = (size_t) (convTable.numPoints * convTable.numFeatures);
      convData = (float *) malloc(sizeof(float)*numValues);
      if (convData==NULL) {
         fprintf(stderr, "%s \"%s\": unable to allocate memory for loading dataFile \"%s\": %s.\n",
               getKeyword(), getName(), dataFile, strerror(errno));
      }
      numRead = fread(convData, sizeof(float), numValues, conversionTableFP);
      if(numRead != numValues) {
         fprintf(stderr, "%s \"%s\" error reading dataFile \"%s\": expecting %zu data values but only read %zu.\n",
               getKeyword(), getName(), dataFile, numValues, numRead);
         exit(EXIT_FAILURE);
      }
      fclose(conversionTableFP);
   }
   MPI_Bcast(&convTable, sizeof(convTableStruct), MPI_CHAR, 0, parent->icCommunicator()->communicator());
   assert(convTable.numFeatures == getLayerLoc()->nf);
   int numValues = convTable.numPoints * convTable.numFeatures;
   if (parent->globalRank()!=0) {
      convData = (float *) malloc(sizeof(float)*numValues);
      if (convData==NULL) {
         fprintf(stderr, "%s \"%s\": unable to allocate memory for loading dataFile \"%s\": %s.\n",
               getKeyword(), getName(), dataFile, strerror(errno));
      }
   }
   MPI_Bcast(convData, numValues, MPI_FLOAT, 0, parent->icCommunicator()->communicator());
   return status;
}

int ConvertFromTable::doUpdateState(double timed, double dt,
      const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels,
      pvdata_t * GSynHead) {
   PVHalo const * halo = &loc->halo;
   float max = convTable.maxRecon;
   float min = convTable.minRecon;
   int numIntervals = convTable.numPoints-1;
   assert(numIntervals > 0);
   // TODO: support for OpenMP and GPU
   for (int k=0; k<getNumNeurons(); k++) {
      int kExt = kIndexExtended(k, loc->nx, loc->ny, loc->nf, halo->lt, halo->rt, halo->dn, halo->up);
      int f = featureIndex(k, loc->nx, loc->ny, loc->nf);
      float * dataAtFeature = &convData[f*convTable.numPoints];
      float Vk = (float) getV()[k];
      float datapoint = (Vk-min)/(max-min)*numIntervals;
      int indexa = floor(datapoint);
      if (indexa < 0) {
         A[kExt] = dataAtFeature[0] + datapoint * (dataAtFeature[1]-dataAtFeature[0]);
      }
      else if (indexa >= numIntervals) {
         A[kExt] = dataAtFeature[numIntervals] + (datapoint-numIntervals) * (dataAtFeature[numIntervals]-dataAtFeature[numIntervals-1]);
      }
      else {
         A[kExt] = dataAtFeature[indexa] + (datapoint-indexa) * (dataAtFeature[indexa+1]-dataAtFeature[indexa]);
      }
   }
   return PV_SUCCESS;
}

ConvertFromTable::~ConvertFromTable() {
   free(dataFile);
   free(convData);
}

