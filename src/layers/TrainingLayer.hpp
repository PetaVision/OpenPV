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
   TrainingLayer(const char * name, HyPerCol * hc, const char * filename);
   virtual ~TrainingLayer();
   int readTrainingLabels(const char * filename, int ** trainingLabels);

   virtual int updateState(float timef, float dt);
   // virtual int updateV();

protected:
   TrainingLayer();
   int initialize(const char * name, HyPerCol * hc, const char * filename);
   int numTrainingLabels;
   int * trainingLabels;
   int curTrainingLabelIndex;
   float displayPeriod;
   float distToData;
   int nextLabelTime;
   pvdata_t strength;

   /* static */ int updateState(float timef, float dt, int numNeurons, pvdata_t * V, int numTrainingLabels, int * trainingLabels, int traininglabelindex, int strength);
   // int setLabeledNeuronToValue(pvdata_t val);
   // int setLabeledNeuron() {return setLabeledNeuronToValue(strength);}
   // int clearLabeledNeuron() {return setLabeledNeuronToValue(0);}
   // void sendBadNeuronMessage();

private:
   int initialize_base();
}; // end class TrainingLayer

}  // end namespace PV block


#endif /* TRAININGLAYER_HPP_ */
