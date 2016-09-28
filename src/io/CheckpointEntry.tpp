/*
 * CheckpointEntry.tpp
 *
 *  Created on Sep 27, 2016
 *      Author: Pete Schultz
 *  template implementations for CheckpointEntry class hierarchy.
 *  Note that the .hpp includes this .tpp file at the end;
 *  the .tpp file does not include the .hpp file.
 */

#include "io/fileio.hpp"
#include "utils/PVLog.hpp"
#include "utils/PVAssert.hpp"

namespace PV {

template <typename T>
void CheckpointEntryData<T>::write(std::string const& checkpointDirectory, double simTime) const {
   if (getCommunicator()->commRank()==0) {
      std::string path = generatePath(checkpointDirectory, "bin");
      PV_Stream * pvstream = PV_fopen(path.c_str(), "w", isVerifyingWrites());
      int numRead = PV_fwrite(mDataPointer, sizeof(T), mNumValues, pvstream);
      if (numRead != mNumValues) {
         pvError() << "CheckpointEntryData::write: unable to write to \"" << path << "\".\n";
      }
      PV_fclose(pvstream);
      path = generatePath(checkpointDirectory, "txt");
      FileStream txtStream(path.c_str(), std::ios_base::out, isVerifyingWrites());
      TextOutput::print(mDataPointer, mNumValues, txtStream.outStream());
   }
}

template <typename T>
void CheckpointEntryData<T>::read(std::string const& checkpointDirectory, double * simTimePtr) const {
   if (getCommunicator()->commRank()==0) {
      std::string path = generatePath(checkpointDirectory, "bin");
      PV_Stream * pvstream = PV_fopen(path.c_str(), "r", false/*reading doesn't use verifyWrites, but PV_fopen constructor still uses it.*/);
      int numRead = PV_fread(mDataPointer, sizeof(T), mNumValues, pvstream);
      if (numRead != mNumValues) {
         pvError() << "CheckpointData::read: unable to read from \"" << path << "\".\n";         
      }
      PV_fclose(pvstream);
   }
   if(mBroadcastingFlag) {
      MPI_Bcast(mDataPointer, mNumValues*sizeof(T), MPI_CHAR, 0, getCommunicator()->communicator()); // TODO: Pack all MPI_Bcasts into a single broadcast.
   }
}

template <typename T>
void CheckpointEntryData<T>::remove(std::string const& checkpointDirectory) const {
   deleteFile(checkpointDirectory, "bin");
   deleteFile(checkpointDirectory, "txt");
}

template <typename T>
void CheckpointEntryPvp<T>::write(std::string const& checkpointDirectory, double simTime) const {
   PV_Stream * pvstream = nullptr;
   if (getCommunicator()->commRank()==0) {
      std::string path = generatePath(checkpointDirectory, "pvp");
      pvstream = PV_fopen(path.c_str(), "w", isVerifyingWrites());
      int * params = pvp_set_nonspiking_act_params(getCommunicator(), simTime, mLayerLoc, mDataType, 1/*numbands*/);
      pvAssert(params && params[1]==NUM_BIN_PARAMS);
      int status = pvp_write_header(pvstream, getCommunicator(), params, params[1]);
      pvErrorIf(status!=PV_SUCCESS, "CheckpointEntryPvp::write unable to write header to to \"%s\"\n", path.c_str());
      free(params);
   }
   for (int b=0; b<mLayerLoc->nbatch; b++) {
      int nxExt = mLayerLoc->nx + mLayerLoc->halo.lt + mLayerLoc->halo.rt;
      int nyExt = mLayerLoc->ny + mLayerLoc->halo.dn + mLayerLoc->halo.up;
      T * batchElementStart = &mDataPointer[b*nxExt*nyExt*mLayerLoc->nf];
      if (pvstream) { PV_fwrite(&simTime, sizeof(double), (size_t) 1, pvstream); }
      gatherActivity(pvstream, getCommunicator(), 0/*rootproc*/, batchElementStart, mLayerLoc, mExtended);
   }
   if (getCommunicator()->commRank()==0) {
      PV_fclose(pvstream);
   }
}

template <typename T>
void CheckpointEntryPvp<T>::read(std::string const& checkpointDirectory, double * simTimePtr) const {
   std::string path = generatePath(checkpointDirectory, "pvp");
   PV_Stream * pvstream = pvp_open_read_file(path.c_str(), getCommunicator());
   int rank = getCommunicator()->commRank();
   pvAssert( (pvstream != nullptr && rank == 0) || (pvstream == nullptr && rank != 0) );
   int numParams = NUM_BIN_PARAMS;
   int params[NUM_BIN_PARAMS];
   int status = pvp_read_header(pvstream, getCommunicator(), params, &numParams);
   if (status != PV_SUCCESS) {
      read_header_err(path.c_str(), getCommunicator(), numParams, params);
   }
   MPI_Datatype * exchangeDatatypes = getCommunicator()->newDatatypes(mLayerLoc);
   std::vector<MPI_Request> req{};
   for (int b=0; b<mLayerLoc->nbatch; b++) {
      int nxExt = mLayerLoc->nx + mLayerLoc->halo.lt + mLayerLoc->halo.rt;
      int nyExt = mLayerLoc->ny + mLayerLoc->halo.dn + mLayerLoc->halo.up;
      T * batchElementStart = &mDataPointer[b*nxExt*nyExt*mLayerLoc->nf];
      if (pvstream) { PV_fread(simTimePtr, sizeof(double), (size_t) 1, pvstream); }
      scatterActivity(pvstream, getCommunicator(), 0/*rootproc*/, batchElementStart, mLayerLoc, mExtended);
      getCommunicator()->exchange(batchElementStart, exchangeDatatypes, mLayerLoc, req);
      // TODO: scattering should be aware of interprocess overlap region so that exchange call isn't necessary.
      getCommunicator()->wait(req);
   }
   getCommunicator()->freeDatatypes(exchangeDatatypes);
   MPI_Bcast(simTimePtr, 1, MPI_DOUBLE, 0, getCommunicator()->communicator());
}

template <typename T>
void CheckpointEntryPvp<T>::remove(std::string const& checkpointDirectory) const {
   deleteFile(checkpointDirectory, "pvp");
}

}