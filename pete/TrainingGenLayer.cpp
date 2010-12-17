/*
 * TrainingGenLayer.cpp
 *
 *  Created on: Dec 8, 2010
 *      Author: pschultz
 */

#include <assert.h>
#include <errno.h>

#include "TrainingGenLayer.hpp"

namespace PV {

TrainingGenLayer::TrainingGenLayer(const char * name, HyPerCol * hc, int numTrainingLabels, int * trainingLabels, float displayPeriod, float delay)
        : GenerativeLayer(name, hc) {
    initialize( numTrainingLabels, trainingLabels, displayPeriod, delay ); // trainingLabels allocated within this initialize call
}  // end of TrainingGenLayer::TrainingGenLayer(const char *, HyPerCol *, int, int *, float, float)

TrainingGenLayer::TrainingGenLayer(const char * name, HyPerCol * hc, const char * filename, float displayPeriod, float delay )
        : GenerativeLayer(name, hc) {
    initialize( filename, displayPeriod, delay ); // trainingLabels allocated within this initialize call
}

TrainingGenLayer::~TrainingGenLayer() {
    free(trainingLabels);
}

int TrainingGenLayer::initialize(int numTrainingLabels, int * trainingLabels, float displayPeriod, float delay) {
    setFuncs(NULL, NULL);
	this->numTrainingLabels = numTrainingLabels;
    this->trainingLabels = NULL;
    curTrainingLabelIndex = 0;
    this->displayPeriod = displayPeriod;
    this->delay = delay;
    this->trainingLabels = (int *) malloc( (size_t) numTrainingLabels * sizeof(int) ); // malloc balanced by free() in destructor
    this->nextLabelTime = displayPeriod + delay;
    if( this->trainingLabels == NULL) return EXIT_FAILURE;

    for( int k=0; k<numTrainingLabels; k++ ) {
    	this->trainingLabels[k] = trainingLabels[k];
    }
    pvdata_t * V = getV();
    for( int k=0; k < getNumNeurons(); k++ ) V[k] = 0;
    // above line not necessary if V was allocated with calloc
    setLabeledNeuron();

    return EXIT_SUCCESS;
}  // end of TrainingGenLayer::initialize(int, int *, float, float)

int TrainingGenLayer::initialize(const char * filename, float displayPeriod, float delay) {
    int * trainingLabelsFromFile;
    int numberOfTrainingLabels = readTrainingLabels( filename, &trainingLabelsFromFile ); // trainingLabelsFromFile allocated within this readTrainingLabels call
    initialize( numberOfTrainingLabels, trainingLabelsFromFile, displayPeriod, delay); // trainingLabels allocated within this initialize call
    free(trainingLabelsFromFile);
    return EXIT_SUCCESS;
}

int TrainingGenLayer::readTrainingLabels(const char * filename, int ** trainingLabelsFromFile) {
    FILE * instream = fopen(filename, "r");
    if( instream == NULL ) {
        fprintf( stderr, "TrainingGenLayer: Unable to open \"%s\". Error %d\n", name, errno );
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
        }
    } while( didReadLabel != EOF );
    fclose(instream);
    *trainingLabelsFromFile = labels;
    return n;
}

int TrainingGenLayer::updateState(float time, float dt) {
    if( time < nextLabelTime ) return EXIT_SUCCESS;

    int status1 = clearLabeledNeuron();

    nextLabelTime += displayPeriod;
    curTrainingLabelIndex++;
    curTrainingLabelIndex = curTrainingLabelIndex == numTrainingLabels ? 0 : curTrainingLabelIndex;
    int status2 = setLabeledNeuron();
    return (status1==EXIT_SUCCESS && status2==EXIT_SUCCESS) ?
           EXIT_SUCCESS : EXIT_FAILURE;
}  // end of TrainingGenLayer::updateState(float, float)

int TrainingGenLayer::setLabeledNeuronToValue(pvdata_t val) {
    int n = trainingLabels[curTrainingLabelIndex];
    int N = getNumNeurons();
    if( n>=N ) {
        sendBadNeuronMessage();
        return EXIT_FAILURE;
    }
    else {
        pvdata_t * V = getV();
        V[trainingLabels[curTrainingLabelIndex]] = val;
        return EXIT_SUCCESS;
    }
}  // end of TrainingGenLayer::setSingleNeuronToValue(int, pvdata_t)

void TrainingGenLayer::sendBadNeuronMessage() {
    fprintf(stderr, "TrainingGenLayer \"%s\":\n", name);
    fprintf(stderr, "Number of training labels is %d\n", numTrainingLabels);
    fprintf(stderr, "Current label index is %d\n", curTrainingLabelIndex);
    fprintf(stderr, "Value of label %d is %d\n", curTrainingLabelIndex,
    		trainingLabels[curTrainingLabelIndex]);
    fprintf(stderr, "Number of neurons is %d\n", getNumNeurons());
}

}  // end of namespace PV
