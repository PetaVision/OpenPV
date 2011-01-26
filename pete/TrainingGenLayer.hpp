/*
 * TrainingGenLayer.hpp
 *
 *  Created on: Dec 8, 2010
 *      Author: pschultz
 */

#ifndef TRAININGGENLAYER_HPP_
#define TRAININGGENLAYER_HPP_

#include "GenerativeLayer.hpp"

namespace PV {

class TrainingGenLayer : public GenerativeLayer {

public:
	TrainingGenLayer(const char * name, HyPerCol * hc, int numTrainingLabels, int * trainingLabels, float displayPeriod, float delay );
	TrainingGenLayer(const char * name, HyPerCol * hc, const char * filename, float displayPeriod, float delay );
	TrainingGenLayer(const char * name, HyPerCol * hc, const char * filename);
	virtual ~TrainingGenLayer();
    int initialize(int numTrainingLabels, int * trainingLabels, float displayPeriod, float delay);
    int initialize(const char * filename, float displayPeriod, float delay);
    int initialize(const char * filename, PVParams * params);
    int readTrainingLabels(const char * filename, int ** trainingLabels);

    virtual int updateState(float time, float dt);

protected:
    int numTrainingLabels;
    int * trainingLabels;
    int curTrainingLabelIndex;
    float displayPeriod;
    float delay;
    int nextLabelTime;

    int setLabeledNeuronToValue(pvdata_t val);
    int setLabeledNeuron() {return setLabeledNeuronToValue(1.0f);}
    int clearLabeledNeuron() {return setLabeledNeuronToValue(0);}
    void sendBadNeuronMessage();
}; // end class TrainingGenLayer

}  // end namespace PV block


#endif /* TRAININGGENLAYER_HPP_ */
