/*
 * TrainingLayer.hpp
 *
 *  Created on: Dec 8, 2010
 *      Author: pschultz
 */

#ifndef TRAININGLAYER_HPP_
#define TRAININGLAYER_HPP_

#include "GenerativeLayer.hpp"

namespace PV {

class TrainingLayer : public ANNLayer {

public:
	TrainingLayer(const char * name, HyPerCol * hc, int numTrainingLabels, int * trainingLabels, float displayPeriod, float delay );
	TrainingLayer(const char * name, HyPerCol * hc, const char * filename, float displayPeriod, float delay );
	TrainingLayer(const char * name, HyPerCol * hc, const char * filename);
	virtual ~TrainingLayer();
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
}; // end class TrainingLayer

}  // end namespace PV block


#endif /* TRAININGLAYER_HPP_ */
