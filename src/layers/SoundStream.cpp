/*
 * SoundStream.cpp
 *
 *  Created on: May 6, 2013
 *      Author: slundquist 
 */

//Only compile this file and its cpp if using sound sandbox
#ifdef PV_USE_SNDFILE

#include "SoundStream.hpp"

#include <stdio.h>

namespace PV {

SoundStream::SoundStream(){
    initialize_base();
}

SoundStream::SoundStream(const char * name, HyPerCol * hc){
    initialize_base();
    initialize(name, hc);
}

SoundStream::~SoundStream() {
    filename = NULL;
	//Communicator::freeDatatypes(mpi_datatypes); mpi_datatypes = NULL;
    //if (getParent()->icCommunicator()->commRank()==0 && fileStream != NULL && fileStream->isfile) {
    //  PV_fclose(fileStream);
    //}
}

int SoundStream::initialize_base() {
	displayPeriod = 1;
	nextDisplayTime = 1;
	filename = NULL;
    soundData = NULL;

	return PV_SUCCESS;
}

int SoundStream::initialize(const char * name, HyPerCol * hc) {
	int status = PV_SUCCESS;
	HyPerLayer::initialize(name, hc, 0);

	free(clayer->V);
	clayer->V = NULL;

	// Point to clayer data struct
    soundData = clayer->activity->data;
    assert(soundData!=NULL);

	//// Create mpi_datatypes for border transfer
	//mpi_datatypes = Communicator::newDatatypes(getLayerLoc());

	//// Exchange border information
	//parent->icCommunicator()->exchange(textData, mpi_datatypes, this->getLayerLoc());

	assert(filename!=NULL);
    filename = strdup(filename);
    assert(filename != NULL);
    fileHeader = new SF_INFO();
    fileStream = sf_open(filename, SFM_READ, fileHeader);
    assert(fileStream != NULL);
    std::cout << "SOUNDSTREAM INITIALIZE!\n";
	//if( getParent()->icCommunicator()->commRank()==0 ) { // Only rank 0 should open the file pointer
	//	filename = strdup(filename);
	//	assert(filename!=NULL );

	//	fileStream = PV_fopen(filename, "r");
	//	if( fileStream->fp == NULL ) {
	//		fprintf(stderr, "TextStream::initialize error opening \"%s\": %s\n", filename, strerror(errno));
	//		status = PV_FAILURE;
	//		abort();
	//	}

	//	// Nav to offset if specified
	//	if (textOffset > 0) {
	//		status = PV_fseek(fileStream,textOffset,SEEK_SET);
	//	}
	//}

	nextDisplayTime = hc->simulationTime();

	status = updateState(0,hc->getDeltaTime());

	return status;
}

int SoundStream::setParams(PVParams * params){
    readSoundInputPath(params);

	int status = HyPerLayer::setParams(params);
    return status;

}

void SoundStream::readSoundInputPath(PVParams * params) {
	filename = params->stringValue(name,"soundInputPath",NULL);
}

int SoundStream::updateState(double time, double dt){
    int status = PV_SUCCESS;
    return status;
}

}

#endif /* PV_USE_SNDFILE */
