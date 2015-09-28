/*
 * HeatMapProbe.cpp
 *
 *  Created on: Sep 23, 2015
 *      Author: pschultz
 */

#include <sstream>
#include <sys/wait.h>
#include "HeatMapProbe.hpp"
#include "parseConfigFile.hpp"

HeatMapProbe::HeatMapProbe(const char * probeName, PV::HyPerCol * hc) {
   initialize_base();
   initialize(probeName, hc);
}

HeatMapProbe::HeatMapProbe() {
   initialize_base();
}

int HeatMapProbe::initialize_base() {
   imageLayerName = NULL;
   resultLayerName = NULL;
   reconLayerName = NULL;
   resultTextFile = NULL;
   octaveCommand = NULL;
   octaveLogFile = NULL;
   classNames = NULL;
   evalCategoryIndices = NULL;
   displayCategoryIndices = NULL;
   highlightThreshold = NULL;
   heatMapThreshold = NULL;
   heatMapMaximum = NULL;
   drawBoundingBoxes = NULL;
   boundingBoxThickness = NULL;
   dbscanEps = NULL;
   dbscanDensity = NULL;
   heatMapMontageDir = NULL;
   displayCommand = NULL;

   imageLayer = NULL;
   resultLayer = NULL;
   reconLayer = NULL;

   outputPeriod = 1.0;
   nextOutputTime = 0.0; // Warning: this does not get checkpointed but it should.  Probes have no checkpointing infrastructure yet.
   octavePid = (pid_t) 0;
   imageLayerName = NULL;
   imageLayer = NULL;
   resultLayerName = NULL;
   resultLayer = NULL;
   reconLayerName = NULL;
   reconLayer = NULL;
   return PV_SUCCESS;
}

int HeatMapProbe::initialize(const char * probeName, PV::HyPerCol * hc) {
   outputPeriod = hc->getDeltaTimeBase(); // default outputPeriod is every timestep
   int status = PV::ColProbe::initialize(probeName, hc);
   PV::InterColComm * icComm = parent->icCommunicator();
   status = parseConfigFile(icComm, &imageLayerName, &resultLayerName, &resultTextFile, &octaveCommand, &octaveLogFile, &classNames, &evalCategoryIndices, &displayCategoryIndices, &highlightThreshold, &heatMapThreshold, &heatMapMaximum, &drawBoundingBoxes, &boundingBoxThickness, &dbscanEps, &dbscanDensity, &heatMapMontageDir, &displayCommand);
   if (status != PV_SUCCESS) { exit(EXIT_FAILURE); }
   return status;
}

int HeatMapProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = PV::ColProbe::ioParamsFillGroup(ioFlag);
   ioParam_reconLayer(ioFlag);
   ioParam_outputPeriod(ioFlag);
   return status;
}

void HeatMapProbe::ioParam_reconLayer(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "reconLayer", &reconLayerName);
}

void HeatMapProbe::ioParam_outputPeriod(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(this->getName(), "triggerLayer"));
   if (!triggerLayer) {
      this->getParent()->ioParamValue(ioFlag, this->getName(), "outputPeriod", &outputPeriod, outputPeriod, true/*warnIfAbsent*/);
   }
}

int HeatMapProbe::initNumValues() {
   return setNumValues(0);
}

int HeatMapProbe::communicateInitInfo() {
   int status = PV::ColProbe::communicateInitInfo();
   PV::HyPerLayer * imageHyPerLayer = parent->getLayerFromName(imageLayerName);
   imageLayer = dynamic_cast<PV::ImageFromMemoryBuffer *>(imageHyPerLayer);
   if (imageLayer==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: imageLayer \"%s\" does not refer to an ImageFromMemoryBuffer in the column.\n",
               name, getKeyword(), imageLayerName);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   resultLayer = parent->getLayerFromName(resultLayerName);
   if (resultLayer==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: resultLayer \"%s\" does not refer to a layer in the column.\n",
               name, getKeyword(), resultLayerName);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   reconLayer = parent->getLayerFromName(reconLayerName);
   if (reconLayer==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: reconLayer \"%s\" does not refer to a layer in the column.\n",
               name, getKeyword(), reconLayerName);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   if (parent->columnId()==0) {
      // clobber octave logfile and result text file unless starting from a checkpoint
      if (parent->getCheckpointReadDir()==NULL) {
         FILE * octaveFP = fopen(octaveLogFile, "w");
         if (octaveFP == NULL) {
            fprintf(stderr, "%s \"%s\": unable to open octave log file \"%s\": %s\n",
                  getKeyword(), getName(), octaveLogFile, strerror(errno));
            status = PV_FAILURE;
         }
         else {
            fclose(octaveFP); // The octave command will write to the octave log file; the probe won't do so directly.
         }
         if (resultTextFile) {
            FILE * resultTextFP = fopen(resultTextFile, "w");
            if (resultTextFP == NULL) {
               fprintf(stderr, "%s \"%s\": unable to open result text file \"%s\": %s\n",
                     getKeyword(), getName(), resultTextFile, strerror(errno));
               status = PV_FAILURE;
            }
            else {
               fclose(resultTextFP); // The octave command will write to the octave log file; the probe won't do so directly.
            }
         }
      }

      // Make the heatmap montage directory if it doesn't already exist.
      struct stat heatMapMontageStat;
      status = stat(heatMapMontageDir, &heatMapMontageStat);
      if (status!=0 && errno==ENOENT) {
         status = mkdir(heatMapMontageDir, 0770);
         if (status!=0) {
            fprintf(stderr, "Error: Unable to make heat map montage directory \"%s\": %s\n", heatMapMontageDir, strerror(errno));
            exit(EXIT_FAILURE);
         }
         status = stat(heatMapMontageDir, &heatMapMontageStat);
      }
      if (status!=0) {
         fprintf(stderr, "Error: Unable to get status of heat map montage directory \"%s\": %s\n", heatMapMontageDir, strerror(errno));
         exit(EXIT_FAILURE);
      }
      if (!(heatMapMontageStat.st_mode & S_IFDIR)) {
         fprintf(stderr, "Error: Heat map montage \"%s\" is not a directory\n", heatMapMontageDir);
         exit(EXIT_FAILURE);
      }

   }

   return status;
}

bool HeatMapProbe::needUpdate(double timed, double dt) {
   bool updateNeeded = false;
   if (triggerLayer) {
      updateNeeded = ColProbe::needUpdate(timed, dt);
   }
   else {
      if (timed>=nextOutputTime) {
         nextOutputTime += outputPeriod;
         updateNeeded = true;
      }
      else {
         updateNeeded = false;
      }
   }
   return updateNeeded;
}

int HeatMapProbe::outputState(double timevalue) {
   // Need to write layers to pvp files in montage directory
   // If octavepid>0, block until it finishes
   // Then fork and and run heatMapMontage.
   // A better way, but one that would require more programming,
   // would be to fork the octave process off during communication.
   // The forked process would monitor for new files and
   // generate montages of them.  If the child process launches
   // a bash script and the bash script launches Octave,
   // the main process could terminate the octave process
   // during its constructor by sending a sigterm to the
   // bash process.
   //
   // I'd really like to have the program deliver data
   // directly from PV to an octave process launched
   // by PV.  How do you do that apart from either sending
   // commands or providing the input in a file?
   // Alternately, just do away with octave and
   // do all the compositing in PetaVision.  Can ImageMagick
   // operate on memory buffers?

   PV::InterColComm * icComm = parent->icCommunicator();
   pvadata_t const * A = NULL;
   PVLayerLoc const * loc = NULL;

   imagePVPFilePath.str("");
   imagePVPFilePath.clear();
   imagePVPFilePath << heatMapMontageDir <<  "/" << "image" << parent->getCurrentStep() << ".pvp";
   A = imageLayer->getLayerData();
   loc = imageLayer->getLayerLoc();
   writeBufferFile(imagePVPFilePath.str().c_str(), icComm, timevalue, A, loc);

   resultPVPFilePath.str("");
   resultPVPFilePath.clear();
   resultPVPFilePath << heatMapMontageDir << "/" << "result" << parent->getCurrentStep() << ".pvp";
   A = resultLayer->getLayerData();
   loc = resultLayer->getLayerLoc();
   writeBufferFile(resultPVPFilePath.str().c_str(), icComm, timevalue, A, loc);

   reconPVPFilePath.str("");
   reconPVPFilePath.clear();
   reconPVPFilePath << heatMapMontageDir << "/" << "recon" << parent->getCurrentStep() << ".pvp";
   A = reconLayer->getLayerData();
   loc = reconLayer->getLayerLoc();
   writeBufferFile(reconPVPFilePath.str().c_str(), icComm, timevalue, A, loc);

   int rank = parent->columnId();
   if (rank==0) {
      if (octavePid>0)
      {
         waitOctaveFinished();
         assert(octavePid==0);
         int waitstatus;
         int waitprocess = waitpid(octavePid, &waitstatus, 0);
         if (waitprocess < 0 && errno != ECHILD)
         {
            fprintf(stderr, "waitpid failed returning %d: %s (%d)\n", waitprocess, strerror(errno), errno);
            exit(EXIT_FAILURE);
         }
         octavePid = 0;
      }
      fflush(NULL); // so that unflushed buffer isn't copied to child process
      octavePid = fork();
      if (octavePid < 0)
      {
         fprintf(stderr, "fork() error: %s\n", strerror(errno));
         exit(EXIT_FAILURE);
      }
      else if (octavePid==0) {
         /* This is the new process - launch Octave and then exit */
         octaveProcess();
         exit(EXIT_SUCCESS);
      }
      else {
         /* Calling process; nothing to do here. */
      }
   }

   return 0;
}

int HeatMapProbe::waitOctaveFinished() {
   int waitstatus;
   int waitprocess = waitpid(octavePid, &waitstatus, 0);
   if (waitprocess < 0 && errno != ECHILD)
   {
      fprintf(stderr, "waitpid failed returning %d: %s (%d)\n", waitprocess, strerror(errno), errno);
      exit(EXIT_FAILURE);
   }
   octavePid = 0;
   return PV_SUCCESS;
}

int HeatMapProbe::octaveProcess() {
   std::stringstream heatMapMontagePath("");
   heatMapMontagePath << heatMapMontageDir << "/heatMap" << parent->getCurrentStep() << ".png";
   std::stringstream octavecommandstream("");
   octavecommandstream << octaveCommand <<
         " --eval 'load CurrentModel/ConfidenceTables/confidenceTable.mat; heatMapMontage(" <<
         "\"" << imagePVPFilePath.str() << "\"" << ", " <<
         "\"" << resultPVPFilePath.str() << "\"" << ", " <<
         "\"" << PV_DIR << "/mlab/util" << "\"" << ", " <<
         1/*imageFrameNumber*/ << ", " <<
         1/*resultFrameNumber*/ << ", " <<
         "confidenceTable, " <<
         "\"" << classNames << "\"" << ", " <<
         "\"" << resultTextFile << "\"" << ", " <<
         evalCategoryIndices << ", " <<
         displayCategoryIndices << ", " <<
         highlightThreshold << ", " <<
         heatMapThreshold << ", " <<
         heatMapMaximum << ", " <<
         drawBoundingBoxes << ", " <<
         boundingBoxThickness << ", " <<
         dbscanEps << ", " <<
         dbscanDensity << ", " <<
         "\"" << heatMapMontagePath.str() << "\"" << ", " <<
         "\"" << displayCommand << "\"" <<
         ");'" <<
         " >> " << octaveLogFile << " 2>&1";
   std::ofstream octavelogstream;
   octavelogstream.open(octaveLogFile, std::fstream::out | std::fstream::app);
   octavelogstream << "Calling octave with the command\n";
   octavelogstream << octavecommandstream.str() << "\n";
   octavelogstream.close();
   int systemstatus = system(octavecommandstream.str().c_str()); // Analysis of the result of the current frame
   octavelogstream.open(octaveLogFile, std::fstream::out | std::fstream::app);
   octavelogstream << "Octave heatMapMontage command returned " << systemstatus << "\n";
   octavelogstream.close();
   return systemstatus;
}

int HeatMapProbe::writeBufferFile(const char * filename, PV::InterColComm * comm, double timevalue, pvadata_t const * A, PVLayerLoc const * loc) {
   PV_Stream * writeFile = pvp_open_write_file(filename, comm, /*append*/false);
   assert( (writeFile != NULL && comm->commRank() == 0) || (writeFile == NULL && comm->commRank() != 0) );

   //nbands gets multiplied by loc->nbatches in this function
   int * params = pvp_set_nonspiking_act_params(comm, timevalue, loc, PV_FLOAT_TYPE, 1);
   assert(params && params[1]==NUM_BIN_PARAMS);
   int status = pvp_write_header(writeFile, comm, params, NUM_BIN_PARAMS);
   if (status != PV_SUCCESS) {
      fprintf(stderr, "HyPerLayer::writeBufferFile error writing \"%s\"\n", filename);
      abort();
   }

   for(int b = 0; b < loc->nbatch; b++){
      if (writeFile != NULL) { // Root process has writeFile set to non-null; other processes to NULL.
         int numwritten = PV::PV_fwrite(&timevalue, sizeof(double), 1, writeFile);
         if (numwritten != 1) {
            fprintf(stderr, "HyPerLayer::writeBufferFile error writing timestamp to \"%s\"\n", filename);
            abort();
         }
      }
      pvadata_t const * bufferBatch;
      bufferBatch = A + b * (loc->nx + loc->halo.rt + loc->halo.lt) * (loc->ny + loc->halo.up + loc->halo.dn) * loc->nf;

      status = gatherActivity(writeFile, comm, 0, bufferBatch, loc);
   }
   free(params);
   pvp_close_file(writeFile, comm);
   writeFile = NULL;
   return status;
}

int HeatMapProbe::gatherActivity(PV_Stream * pvstream, PV::Communicator * comm, int rootproc, pvadata_t const * buffer, const PVLayerLoc * layerLoc) {
   // In MPI when this process is called, all processes must call it.
   // Only the root process uses the file pointer.
   int status = PV_SUCCESS;

   int numLocalNeurons = layerLoc->nx * layerLoc->ny * layerLoc->nf;

   int xLineStart = 0;
   int yLineStart = 0;
   int xBufSize = layerLoc->nx;
   int yBufSize = layerLoc->ny;
   PVHalo halo;
   memcpy(&halo,&layerLoc->halo,sizeof(halo));
   xLineStart = halo.lt;
   yLineStart = halo.up;
   xBufSize += halo.lt+halo.rt;
   yBufSize += halo.dn+halo.up;

   int linesize = layerLoc->nx*layerLoc->nf; // All values across x and f for a specific y are contiguous; do a single write for each y.
   size_t datasize = sizeof(pvadata_t);
   // read into a temporary buffer since buffer may be extended but the file only contains the restricted part.
   pvadata_t * temp_buffer = (pvadata_t *) calloc(numLocalNeurons, datasize);
   if (temp_buffer==NULL) {
      fprintf(stderr, "gatherActivity unable to allocate memory for temp_buffer.\n");
      status = PV_FAILURE;
      abort();
   }

   int rank = comm->commRank();
   if (rank==rootproc) {
      if (pvstream == NULL) {
         fprintf(stderr, "gatherActivity error: file pointer on root process is null.\n");
         status = PV_FAILURE;
         abort();
      }
      long startpos = PV::getPV_StreamFilepos(pvstream);
      if (startpos == -1) {
         fprintf(stderr, "gatherActivity error when getting file position: %s\n", strerror(errno));
         status = PV_FAILURE;
         abort();
      }
      // Write zeroes to make sure the file is big enough since we'll write nonsequentially under MPI.  This may not be necessary.
      int comm_size = comm->commSize();
      for (int r=0; r<comm_size; r++) {
         int numwritten = PV::PV_fwrite(temp_buffer, datasize, numLocalNeurons, pvstream);
         if (numwritten != numLocalNeurons) {
            fprintf(stderr, "gatherActivity error when writing: number of bytes attempted %d, number written %d\n", numwritten, numLocalNeurons);
            status = PV_FAILURE;
            abort();
         }
      }
      int fseekstatus = PV::PV_fseek(pvstream, startpos, SEEK_SET);
      if (fseekstatus != 0) {
         fprintf(stderr, "gatherActivity error when setting file position: %s\n", strerror(errno));
         status = PV_FAILURE;
         abort();
      }

      for (int r=0; r<comm_size; r++) {
         if (r==rootproc) {
            for (int y=0; y<layerLoc->ny; y++) {
               int k_extended = kIndex(halo.lt, y+yLineStart, 0, xBufSize, yBufSize, layerLoc->nf);
               int k_restricted = kIndex(0, y, 0, layerLoc->nx, layerLoc->ny, layerLoc->nf);
               memcpy(&temp_buffer[k_restricted], &buffer[k_extended], datasize*linesize);
            }
         }
         else {
            MPI_Recv(temp_buffer, numLocalNeurons*(int) datasize, MPI_BYTE, r, 171+r/*tag*/, comm->communicator(), MPI_STATUS_IGNORE);
         }

         // Data to be written is in temp_buffer, which is nonextend.
         for (int y=0; y<layerLoc->ny; y++) {
            int ky0 = layerLoc->ny*rowFromRank(r, comm->numCommRows(), comm->numCommColumns());
            int kx0 = layerLoc->nx*columnFromRank(r, comm->numCommRows(), comm->numCommColumns());
            int k_local = kIndex(0, y, 0, layerLoc->nx, layerLoc->ny, layerLoc->nf);
            int k_global = kIndex(kx0, y+ky0, 0, layerLoc->nxGlobal, layerLoc->nyGlobal, layerLoc->nf);
            int fseekstatus = PV::PV_fseek(pvstream, startpos + k_global*datasize, SEEK_SET);
            if (fseekstatus == 0) {
               int numwritten = PV::PV_fwrite(&temp_buffer[k_local], datasize, linesize, pvstream);
               if (numwritten != linesize) {
                  fprintf(stderr, "gatherActivity error when writing to \"%s\": number of bytes attempted %zu, number written %d\n", pvstream->name, datasize*linesize, numwritten);
                  status = PV_FAILURE;
               }
            }
            else {
               fprintf(stderr, "gatherActivity error when setting file position: %s\n", strerror(errno));
               status = PV_FAILURE;
               abort();
            }
         }
      }
      PV::PV_fseek(pvstream, startpos+numLocalNeurons*datasize*comm_size, SEEK_SET);
   }
   else {
      if (halo.lt || halo.rt || halo.dn || halo.up) {
         // temp_buffer is a restricted buffer, but if extended is true, buffer is an extended buffer.
         for (int y=0; y<layerLoc->ny; y++) {
            int k_extended = kIndex(halo.lt, y+yLineStart, 0, xBufSize, yBufSize, layerLoc->nf);
            int k_restricted = kIndex(0, y, 0, layerLoc->nx, layerLoc->ny, layerLoc->nf);
            memcpy(&temp_buffer[k_restricted], &buffer[k_extended], datasize*linesize);
         }
         MPI_Send(temp_buffer, numLocalNeurons*datasize, MPI_BYTE, rootproc, 171+rank/*tag*/, comm->communicator());
      }
      else {
         MPI_Send(buffer, numLocalNeurons*datasize, MPI_BYTE, rootproc, 171+rank/*tag*/, comm->communicator());
      }
   }

   free(temp_buffer); temp_buffer = NULL;
   return status;
}

HeatMapProbe::~HeatMapProbe() {
   free(imageLayerName);
   free(resultLayerName);
   free(reconLayerName);
}

