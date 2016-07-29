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

int ConvertFromTable::ioParamsFillGroup(enum PV::ParamsIOFlag ioFlag) {
   int status = PV::CloneVLayer::ioParamsFillGroup(ioFlag);
   ioParam_dataFile(ioFlag);
   return status;
}

void ConvertFromTable::ioParam_dataFile(enum PV::ParamsIOFlag ioFlag) {
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
         pvError().printf("%s: unable to open dataFile \"%s\": %s\n",
               getDescription_c(), dataFile, strerror(errno));
      }
      status = fseek(conversionTableFP, 0L, SEEK_END);
      if (status != 0) {
         pvError().printf("%s: unable to seek to end off dataFile \"%s\": %s\n",
               getDescription_c(), dataFile, strerror(errno));
      }
      long fileLength = ftell(conversionTableFP);
      if (fileLength<0) {
         pvError().printf("%s: unable to get length of dataFile \"%s\": %s\n",
               getDescription_c(), dataFile, strerror(errno));
      }
      if (fileLength==0) {
         pvError().printf("%s: dataFile \"%s\" is empty.\n",
               getDescription_c(), dataFile);
      }
      if (fileLength<CONVTABLEHEADERSIZE) {
         pvError().printf("%s \"%s\": dataFile \"%s\" is too small (%ld bytes; minimum is %zu bytes)\n",
               getDescription_c(), dataFile, fileLength, CONVTABLEHEADERSIZE);
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
         pvError().printf("%s \"%s\": reading header of dataFile \"%s\": expected %zu bytes; only read %zu\n",
               getDescription_c(), dataFile, CONVTABLEHEADERSIZE, numRead);
      }
      if (memcmp(hdr, "convTabl", 8)) {
         pvError().printf("%s \"%s\": dataFile \"%s\" does not have the expected header.\n",
               getDescription_c(), dataFile);
      }
      int correctFileSize = (int) (sizeof(float)*convTable.numPoints*convTable.numFeatures+CONVTABLEHEADERSIZE);
      if (correctFileSize != fileLength) {
         pvError().printf("%s: dataFile \"%s\" has unexpected length (%d-by-%d %zu-byte data values with %zu-byte header, but file size is %ld\n",
               getDescription_c(), dataFile, convTable.numPoints, convTable.numFeatures, sizeof(float), CONVTABLEHEADERSIZE, fileLength);
      }
      if (convTable.numFeatures != getLayerLoc()->nf) {
         pvError().printf("%s: layer has %d features but dataFile \"%s\" has numFeatures=%d\n",
               getDescription_c(), getLayerLoc()->nf, dataFile, convTable.numFeatures);
      }
      if (convTable.numFeatures < 2) {
         pvError().printf("%s: dataFile \"%s\" has numPoints=%d but must be >= 2.\n",
               getDescription_c(), dataFile, convTable.numPoints);
      }
      size_t numValues = (size_t) (convTable.numPoints * convTable.numFeatures);
      convData = (float *) malloc(sizeof(float)*numValues);
      if (convData==NULL) {
         pvError().printf("%s: unable to allocate memory for loading dataFile \"%s\": %s.\n",
               getDescription_c(), dataFile, strerror(errno));
      }
      numRead = fread(convData, sizeof(float), numValues, conversionTableFP);
      if(numRead != numValues) {
         pvError().printf("%s: unable to read dataFile \"%s\": expecting %zu data values but only read %zu.\n",
               getDescription_c(), dataFile, numValues, numRead);
      }
      fclose(conversionTableFP);
   }
   MPI_Bcast(&convTable, sizeof(convTableStruct), MPI_CHAR, 0, parent->getCommunicator()->communicator());
   assert(convTable.numFeatures == getLayerLoc()->nf);
   int numValues = convTable.numPoints * convTable.numFeatures;
   if (parent->globalRank()!=0) {
      convData = (float *) malloc(sizeof(float)*numValues);
      if (convData==NULL) {
         pvError().printf("%s: unable to allocate memory for loading dataFile \"%s\": %s.\n",
               getDescription_c(), dataFile, strerror(errno));
      }
   }
   MPI_Bcast(convData, numValues, MPI_FLOAT, 0, parent->getCommunicator()->communicator());
   return status;
}

int ConvertFromTable::updateState(double timed, double dt)
{
   const PVLayerLoc * loc = getLayerLoc();
   pvdata_t * A = clayer->activity->data;
   pvdata_t * V = getV();
   int num_channels = getNumChannels();
   pvdata_t * GSynHead = GSyn == NULL ? NULL : GSyn[0];
   
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

PV::BaseObject * createConvertFromTable(char const * name, PV::HyPerCol * hc) {
   return hc ? new ConvertFromTable(name, hc) : NULL;
}
