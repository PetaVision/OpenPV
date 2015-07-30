/*
 * TrainingLayer.hpp
 *
 * A layer where the V and A levels are determined by a list of training labels
 * The list of labels is specified in the constructor, either by an array of int's
 * or a filename.  The labels are integers from 0 through n-1, where n is the
 * number of neurons in the layer.
 *
 *  Created on: Dec 8, 2010
 *      Author: pschultz
 */

#ifndef TRAININGLAYER_HPP_
#define TRAININGLAYER_HPP_

#include "ANNLayer.hpp"

namespace PV {

class TrainingLayer : public ANNLayer {

public:
   TrainingLayer(const char * name, HyPerCol * hc);
   virtual ~TrainingLayer();
   virtual int allocateDataStructures();
   int readTrainingLabels(const char * filename, int ** trainingLabels);
   virtual bool needUpdate(double timed, double dt);
   virtual int recvAllSynapticInput() {return PV_SUCCESS;}
   virtual int updateState(double timed, double dt);
   virtual int checkpointWrite(const char * cpDir);
   virtual double getDeltaUpdateTime();

protected:
   TrainingLayer();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_trainingLabelsPath(enum ParamsIOFlag ioFlag);
   virtual void ioParam_displayPeriod(enum ParamsIOFlag ioFlag);
   virtual void ioParam_distToData(enum ParamsIOFlag ioFlag);
   virtual void ioParam_strength(enum ParamsIOFlag ioFlag);
   virtual int initializeV();

   /* static */ int updateState(double timed, double dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int numTrainingLabels, int * trainingLabels, int traininglabelindex, int strength);
   int readStateFromCheckpoint(const char * cpDir, double * timeptr);
   int read_currentLabelIndexFromCheckpoint(const char * cpDir);

   char * filename;
   int numTrainingLabels;
   int * trainingLabels;
   int curTrainingLabelIndex;
   double displayPeriod;
   double distToData;
   // int nextLabelTime; // Now handled by HyPerLayer's nextUpdateTime
   pvdata_t strength;

private:
   int initialize_base();
}; // end class TrainingLayer

}  // end namespace PV block


#endif /* TRAININGLAYER_HPP_ */
