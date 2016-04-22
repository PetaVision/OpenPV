/*
 * MapReduceKernelConn.cpp
 *
 *  Created on: Aug 16, 2013
 *      Author: garkenyon
 */

#include "MapReduceKernelConn.hpp"
#include <iostream>

namespace PV {

MapReduceKernelConn::MapReduceKernelConn() {
	initialize_base();
}

MapReduceKernelConn::~MapReduceKernelConn() {
}

int MapReduceKernelConn::initialize_base() {
	num_dWeightFiles = 1;
	dWeightFileIndex = 0;
	//dWeightsList = NULL;
	movieLayerName = NULL;
	movieLayer = NULL;
	return PV_SUCCESS;
}

MapReduceKernelConn::MapReduceKernelConn(const char * name, HyPerCol *hc) {
   MapReduceKernelConn::initialize_base();
   MapReduceKernelConn::initialize(name, hc);
}

int MapReduceKernelConn::initialize(const char * name, HyPerCol * hc) {
   int status = HyPerConn::initialize(name, hc);
   // if (status==PV_SUCCESS) status = setMovieLayerName();
   InterColComm *icComm = parent->icCommunicator();
   int rootproc = 0;
   int file_count = 0;
   PV_Stream * dWeightstream = pvp_open_read_file(this->dWeightsListName,
           icComm);
   if ((dWeightstream == NULL) && (icComm->commRank() == rootproc)) {
       fprintf(stderr,
             "MapReduceKernelConn::initialize: Cannot open list of dWeight files \"%s\".  Exiting.\n",
             this->dWeightsListName); // CHECK THIS! // CHECK THIS! this->dWeightsListName was filename but filename is no more
       exit(EXIT_FAILURE);
   } // dWeightstream == NULL

   if (icComm->commRank() == rootproc) {
      for (file_count = 0; file_count < num_dWeightFiles; file_count++) {
         /*
         for (int i_char = 0; i_char < PV_PATH_MAX; i_char++) {
            dWeightsList[file_count][i_char] = 0;
         }
         */

         char * fgetsstatus = fgets(dWeightsList[file_count], PV_PATH_MAX, dWeightstream->fp);
         if (fgetsstatus == NULL) {
            bool endoffile = feof(dWeightstream->fp) != 0;
            if (endoffile) {
               fprintf(stderr,
                       "MapReduceKernelConn::initialize: "
                               "File of weight files \"%s\" reached end of file before all %d weight files were read.  "
                               "Exiting.\n", dWeightstream->name, num_dWeightFiles); // CHECK THIS! dWeightstream->name was filename but filename is no more
               exit(EXIT_FAILURE);
            }
            else {
               int error = ferror(dWeightstream->fp);
               assert(error);
               fprintf(stderr,
                     "MapReduceKernelConn::initialize: File of weight files error reading:%s.  Exiting.\n", strerror(error));
               exit(error);
            }
         }
         else {
            // Remove linefeed from end of string
            dWeightsList[file_count][PV_PATH_MAX - 1] = '\0';
            int len = strlen(dWeightsList[file_count]);
            if (len > 1) {
               if (dWeightsList[file_count][len - 1] == '\n') {
                  dWeightsList[file_count][len - 1] = '\0';
               }
            }
         } // fgetsstatus == NULL
     } // file_count
     this->dWeightsFilename = strdup(dWeightsList[dWeightFileIndex]);
     std::cout << "dWeightFile[" << dWeightFileIndex << "] = "
           << dWeightsList[dWeightFileIndex] << std::endl;
   } // commRank() == rootproc
   return status;
}

int MapReduceKernelConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerConn::ioParamsFillGroup(ioFlag);
   ioParam_movieLayerName(ioFlag);
   ioParam_dWeightsListName(ioFlag);
   ioParam_num_dWeightFiles(ioFlag);
   ioParam_dWeightFileIndex(ioFlag);
   // TODO::use movie layer to derive dW weight file name and contents of dW list when not passed in directly by user
   return status;
}

// TODO: make sure code works in non-shared weight case
void MapReduceKernelConn::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   sharedWeights = true;
   if (ioFlag == PARAMS_IO_READ) {
      fileType = PVP_KERNEL_FILE_TYPE;
      parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights", true/*correctValue*/);
   }
}

void MapReduceKernelConn::ioParam_movieLayerName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "movieLayerName", &movieLayerName);
}

void MapReduceKernelConn::ioParam_dWeightsListName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "dWeightsListName", &dWeightsListName);
}

void MapReduceKernelConn::ioParam_num_dWeightFiles(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "num_dWeightFiles", &num_dWeightFiles, num_dWeightFiles, true/*warnIfAbsent*/);
   if (ioFlag == PARAMS_IO_READ && num_dWeightFiles <= 0) { // Should the error be on < 0 or should the message say 'greater than zero'?
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\": num_dWeightFiles must be greater than or equal to zero.\n",
               parent->parameters()->groupKeywordFromName(name), name);
      }
#ifdef PV_USE_MPI
      MPI_Barrier(parent->icCommunicator()->communicator());
#endif
      exit(EXIT_FAILURE);
   }
}

void MapReduceKernelConn::ioParam_dWeightFileIndex(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "dWeightFileIndex", &dWeightFileIndex, dWeightFileIndex, true/*warnIfAbsent*/);
   if (dWeightFileIndex < 0) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\": dWeightFileIndex must be greater than zero.\n",
               parent->parameters()->groupKeywordFromName(name), name);
      }
#ifdef PV_USE_MPI
      MPI_Barrier(parent->icCommunicator()->communicator());
#endif
      exit(EXIT_FAILURE);
   }
}

int MapReduceKernelConn::communicateInitInfo() {
	int status = HyPerConn::communicateInitInfo();

	HyPerLayer * origHyPerLayer = parent->getLayerFromName(movieLayerName);
	if (origHyPerLayer == NULL) {
		fprintf(stderr,
				"MapReduceKernelConn \"%s\" error: movieLayerName \"%s\" is not a layer in the HyPerCol.\n",
				name, movieLayerName);
		return (EXIT_FAILURE);
	}
	movieLayer = dynamic_cast<Movie *>(origHyPerLayer);
	if (movieLayer == NULL) {
		fprintf(stderr,
				"MapReduceKernelConn \"%s\" error: movieLayerName \"%s\" is not a"
						" Movie or Movie-derived layer in the HyPerCol.\n",
				name, movieLayerName);
		return (EXIT_FAILURE);
	}

	return status;
}

int MapReduceKernelConn::reduceKernels(const int arborID) {
	int status = HyPerConn::reduceKernels(arborID);
	int rootproc = 0;
	InterColComm *icComm = parent->icCommunicator();
	const int numPatches = getNumDataPatches();
	const size_t patchSize = nxp * nyp * nfp * sizeof(pvdata_t);
	const size_t localSize = numPatches * patchSize;
	const size_t arborSize = localSize * this->numberOfAxonalArborLists();
	if (icComm->commRank() == rootproc) {
		// write dW for this instantiation of PetaVision to disk
		status = HyPerConn::writeWeights(NULL, this->get_dwDataStart(),
				getNumDataPatches(), dWeightsList[dWeightFileIndex],
				parent->simulationTime(), /*writeCompressedWeights*/false, /*last*/
				false);
		if (status != PV_SUCCESS) {
			fprintf(stderr,
					"MapReduceKernelConn::reduceKernels::HyPerConn::writeWeights: problem writing to file %s, "
							"SHUTTING DOWN\n", dWeightsList[dWeightFileIndex]);
			exit(EXIT_FAILURE);
		} // status
		  // use dWeightsList to read in the weights written by other PetaVision instantiations
		double dW_time;
		double simulation_time = parent->simulationTime();
		int filetype, datatype;
		int numParams = NUM_BIN_PARAMS + NUM_WGT_EXTRA_PARAMS;
		int params[NUM_BIN_PARAMS + NUM_WGT_EXTRA_PARAMS];
		const PVLayerLoc *preLoc = this->preSynapticLayer()->getLayerLoc();
		int file_count = 0;
		for (file_count = 0; file_count < num_dWeightFiles; file_count++) {
			if (file_count == dWeightFileIndex) {
				continue;
			}
			int num_attempts = 0;
			const int MAX_ATTEMPTS = 5;
			dW_time = 0;
			while (dW_time < simulation_time && num_attempts <= MAX_ATTEMPTS) {
				pvp_read_header(dWeightsList[file_count], icComm, &dW_time,
						&filetype, &datatype, params, &numParams);
				num_attempts++;
			} // while
			if (num_attempts > MAX_ATTEMPTS) {
				fprintf(stderr,
						"PV::MapReduceKernelConn::reduceKernels: problem reading arbor file %s, SHUTTING DOWN\n",
						dWeightsList[file_count]);
				status = EXIT_FAILURE;
				exit(EXIT_FAILURE);
			} // num_attempts > MAX_ATTEMPTS
			int status = PV::readWeights(NULL, get_dwDataStart(),
					this->numberOfAxonalArborLists(), this->getNumDataPatches(), nxp, nyp, nfp,
					dWeightsList[file_count], icComm, &dW_time, preLoc);
			if (status != PV_SUCCESS) {
				fprintf(stderr,
						"MapReduceKernelConn::reduceKernels::PV::readWeights: problem reading file %s, "
								"SHUTTING DOWN\n", dWeightsList[file_count]);
				exit(EXIT_FAILURE);
			} // status
		} // file_count < numWeightFiles
		  // average dW from map-reduce
		pvwdata_t * dW_data = this->get_dwDataStart(0);
		for (int i_dW = 0; i_dW < arborSize; i_dW++) {
			dW_data[i_dW] /= num_dWeightFiles;
		}
	} // rootproc

	// broadcast map-reduced dWeights to all non-root processes
	MPI_Comm mpi_comm = icComm->communicator();
#ifdef PV_USE_MPI
	MPI_Bcast(this->get_wDataStart(0), arborSize, MPI_FLOAT, rootproc, mpi_comm);
#endif
	return PV_BREAK;
}

} /* namespace PV */
