/*
 * TrainingLayer.cpp
 *
 *  Created on: Dec 8, 2010
 *      Author: pschultz
 */

#include "TrainingLayer.hpp"

namespace PV {

TrainingLayer::TrainingLayer(const char * name, HyPerCol * hc, int numTrainingLabels, int * trainingLabels, float displayPeriod, float distToData)
        : ANNLayer(name, hc) {
    initialize( numTrainingLabels, trainingLabels, displayPeriod, distToData ); // trainingLabels allocated within this initialize call
}  // end of TrainingLayer::TrainingLayer(const char *, HyPerCol *, int, int *, float, float)

TrainingLayer::TrainingLayer(const char * name, HyPerCol * hc, const char * filename, float displayPeriod, float distToData )
        : ANNLayer(name, hc) {
    initialize( filename, displayPeriod, distToData ); // trainingLabels allocated within this initialize call
}

TrainingLayer::TrainingLayer(const char * name, HyPerCol * hc, const char * filename)
        : ANNLayer(name, hc) {
    initialize( filename, hc->parameters() );
}

TrainingLayer::~TrainingLayer() {
    free(trainingLabels);
}

int TrainingLayer::initialize(int numTrainingLabels, int * trainingLabels, float displayPeriod, float distToData) {
    // setFuncs(NULL, NULL);
    this->numTrainingLabels = numTrainingLabels;
    this->trainingLabels = NULL;
    curTrainingLabelIndex = 0;
    this->displayPeriod = displayPeriod;
    this->distToData = distToData;
    this->trainingLabels = (int *) malloc( (size_t) numTrainingLabels * sizeof(int) ); // malloc balanced by free() in destructor
    this->nextLabelTime = displayPeriod + distToData;
    if( this->trainingLabels == NULL) return PV_FAILURE;

    for( int k=0; k<numTrainingLabels; k++ ) {
        this->trainingLabels[k] = trainingLabels[k];
    }
    pvdata_t * V = getV();
    for( int k=0; k < getNumNeurons(); k++ ) V[k] = 0;
    // above line not necessary if V was allocated with calloc
    setLabeledNeuron();
    setActivity(); // needed because updateState won't call setActivity until the first update period has passed.

    return PV_SUCCESS;
}  // end of TrainingLayer::initialize(int, int *, float, float)

int TrainingLayer::initialize(const char * filename, float displayPeriod, float distToData) {
    int * trainingLabelsFromFile;
    int numberOfTrainingLabels = readTrainingLabels( filename, &trainingLabelsFromFile ); // trainingLabelsFromFile allocated within this readTrainingLabels call
    initialize( numberOfTrainingLabels, trainingLabelsFromFile, displayPeriod, distToData); // trainingLabels allocated within this initialize call
    free(trainingLabelsFromFile);
    return PV_SUCCESS;
}

int TrainingLayer::initialize(const char * filename, PVParams * params) {
    float displayPeriod = params->value(name, "displayPeriod", -1);
    float distToData = params->value(name, "distToData", -1);
    if( displayPeriod < 0 || distToData < 0) {
    	fprintf(stderr, "Constructor for TrainingLayer \"%s\" requires parameters displayPeriod and distToData to be set to nonnegative values in the params file.\n", name);
        exit(PV_FAILURE);
    }
    return initialize(filename, displayPeriod, distToData);
}

int TrainingLayer::readTrainingLabels(const char * filename, int ** trainingLabelsFromFile) {
    FILE * instream = fopen(filename, "r");
    if( instream == NULL ) {
        fprintf( stderr, "TrainingLayer: Unable to open \"%s\". Error %d\n", filename, errno );
        *trainingLabelsFromFile = NULL;
        return 0;
    }

    int didReadLabel;
    int n = 0;
    int label;
    int * labels = NULL;
    int * oldlabels;
    do {
        didReadLabel = fscanf(instream, "%d", &label);
        switch( didReadLabel ) {
        case 0:
            fseek( instream, 1L, SEEK_CUR );
            break;
        case 1:
            n++;
            oldlabels = labels;
            labels = (int *) malloc((size_t) n * sizeof(int) );
            assert(labels);
            for(int k=0; k<n-1; k++) labels[k] = oldlabels[k];
            labels[n-1] = label;
            free(oldlabels);
            break;
        }
    } while( didReadLabel != EOF );
    fclose(instream);
    *trainingLabelsFromFile = labels;
    return n;
}

int TrainingLayer::updateState(float time, float dt) {
    if(time >= nextLabelTime) {
        nextLabelTime += displayPeriod;
        return ANNLayer::updateState(time, dt);
    }
    else return PV_SUCCESS;
}

int TrainingLayer::updateV() {
    int status1 = clearLabeledNeuron();

    curTrainingLabelIndex++;
    curTrainingLabelIndex = curTrainingLabelIndex == numTrainingLabels ? 0 : curTrainingLabelIndex;
    int status2 = setLabeledNeuron();
    return (status1==PV_SUCCESS && status2==PV_SUCCESS) ?
           PV_SUCCESS : PV_FAILURE;
}  // end of TrainingLayer::updateV()

int TrainingLayer::setLabeledNeuronToValue(pvdata_t val) {
    int n = trainingLabels[curTrainingLabelIndex];
    int N = getNumNeurons();
    if( n>=N ) {
        sendBadNeuronMessage();
        return PV_FAILURE;
    }
    else {
        pvdata_t * V = getV();
        V[trainingLabels[curTrainingLabelIndex]] = val;
        return PV_SUCCESS;
    }
}  // end of TrainingLayer::setLabeledNeuronToValue(int, pvdata_t)

void TrainingLayer::sendBadNeuronMessage() {
    fprintf(stderr, "TrainingLayer \"%s\":\n", name);
    fprintf(stderr, "Number of training labels is %d\n", numTrainingLabels);
    fprintf(stderr, "Current label index is %d\n", curTrainingLabelIndex);
    fprintf(stderr, "Value of label %d is %d\n", curTrainingLabelIndex,
            trainingLabels[curTrainingLabelIndex]);
    fprintf(stderr, "Number of neurons is %d\n", getNumNeurons());
}

}  // end of namespace PV
