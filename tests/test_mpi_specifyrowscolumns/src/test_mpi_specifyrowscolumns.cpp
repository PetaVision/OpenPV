/**
 * This file tests whether the HyPerCol parses
 * the -rows and -columns arguments correctly
 * Use with mpirun -np 6
 *
 */

#include <include/pv_common.h>

#ifndef PV_USE_MPI
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char * argv[]) {
   fprintf(stderr, "%s: this test can only be used under MPI with exactly six processes.\n", argv[0]);
   // TODO Greater than six should be permissible, with the excess over 6 being idle
   exit(EXIT_FAILURE);
}
#else // ifndef PV_USE_MPI

#include "columns/HyPerCol.hpp"
#include "layers/ANNLayer.hpp"
#include "io/io.hpp"
#include "assert.h"
#include "arch/mpi/mpi.h"

int buildandverify(PV::PV_Init* initObj);
int verifyLoc(PV::HyPerCol * loc, int rows, int columns);
int dumpLoc(const PVLayerLoc * loc, int rank);

using namespace PV;

int main(int argc, char * argv[]) {
   PV_Init* initObj = new PV_Init(&argc, &argv, false/*allowUnrecognizedArguments*/);
   int status = PV_SUCCESS;
   int rank = 0;
   int numProcs = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
   //int numProcs = initObj->getWorldSize();

   if( numProcs != 6) {
      // TODO Greater than six should be permissible, with the excess over 6 being idle
      if (rank==0) {
         fprintf(stderr, "%s: this test can only be used under MPI with exactly six processes.\n", argv[0]);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      exit(EXIT_FAILURE);
   }
   
   PV_Arguments * arguments = initObj->getArguments();
   if (arguments->getParamsFile()!=NULL) {
      if (rank==0) {
         fprintf(stderr, "%s should be run without the params file argument.\n", arguments->getProgramName());
      }
      status = PV_FAILURE;
   }
   if (arguments->getNumRows()!=0) {
      if (rank==0) {
         fprintf(stderr, "%s should be run without the rows argument.\n", arguments->getProgramName());
      }
      status = PV_FAILURE;
   }
   if (arguments->getNumColumns()!=0) {
      if (rank==0) {
         fprintf(stderr, "%s should be run without the columns argument.\n", arguments->getProgramName());
      }
      status = PV_FAILURE;
   }
   if (status != PV_SUCCESS) {
      if (rank==0) {
         fprintf(stderr, "The necessary parameters are hardcoded.\n");
      }
      MPI_Barrier(MPI_COMM_WORLD);
      exit(EXIT_FAILURE);
   }

   arguments->setParamsFile("input/test_mpi_specifyrowscolumns.params");
   arguments->setNumRows(2);
   arguments->setNumColumns(3);
   buildandverify(initObj);

   arguments->setNumRows(3);
   arguments->setNumColumns(2);
   buildandverify(initObj);

   delete initObj;
   return status;
}

int buildandverify(PV::PV_Init* initObj) {
   initObj->initialize();
   PV::HyPerCol * hc = new PV::HyPerCol("column", initObj);
   /* PV::ANNLayer * layer = */ new PV::ANNLayer("layer", hc);
   int rows = initObj->getArguments()->getNumRows();
   int columns = initObj->getArguments()->getNumColumns();
   assert(rows > 0 && columns > 0);
   int status = verifyLoc(hc, rows, columns);
   delete hc;
   return status;
}

int verifyLoc(PV::HyPerCol * hc, int rows, int columns) {
   int status = PV_SUCCESS;
   int testpassed;
   const PVLayerLoc * loc = hc->getLayer(0)->getLayerLoc();
   int rank = hc->icCommunicator()->commRank();
   assert(rows == hc->icCommunicator()->numCommRows());
   assert(columns == hc->icCommunicator()->numCommColumns());
   PVParams * params = hc->parameters();
   int nxGlobFromParams = params->value("column", "nx");
   int nyGlobFromParams = params->value("column", "ny");
   testpassed = (loc->nx == nxGlobFromParams/columns) &&
                (loc->ny == nyGlobFromParams/rows) &&
                (loc->nf == params->value("layer", "nf")) &&
                (loc->nxGlobal == nxGlobFromParams) &&
                (loc->nyGlobal == nyGlobFromParams) &&
                (loc->kx0 == loc->nx * (rank % columns)) &&
                (loc->ky0 == loc->ny * (rank / columns));

   PVLayerLoc mpiLoc;
   if( rank == 0 ) {
      fprintf(stdout, "Testing with %d rows by %d columns of subprocesses.\n", rows, columns);
      if( testpassed ) {
         fprintf(stdout, "Rank 0 passed.\n");
      }
      else {
         dumpLoc(loc, 0);
         fflush(stdout);
         fprintf(stderr, "Rank 0 FAILED\n");
         status = PV_FAILURE;
      }
      // Receive each process's testpassed value and output it.
      for( int src=1; src<hc->icCommunicator()->commSize(); src++) {
         int remotepassed;
         MPI_Recv(&remotepassed, 1, MPI_INT, src, 10, hc->icCommunicator()->communicator(), MPI_STATUS_IGNORE);
         if( remotepassed ) {
            fprintf(stderr, "Rank %d passed.\n", src);
         }
         else {
            MPI_Recv(&mpiLoc, sizeof(PVLayerLoc), MPI_CHAR, src, 20, hc->icCommunicator()->communicator(), MPI_STATUS_IGNORE);
            dumpLoc(&mpiLoc, src);
            fflush(stdout);
            fprintf(stderr, "Rank %d FAILED\n", src);
            status = PV_FAILURE;
         }
      }
   }
   else {
      // Send each process's testpassed value to root process.
      MPI_Send(&testpassed, 1, MPI_INT, 0, 10, hc->icCommunicator()->communicator());
      if( !testpassed ) {
         memcpy(&mpiLoc, loc, sizeof(PVLayerLoc));
         MPI_Send(&mpiLoc, sizeof(PVLayerLoc), MPI_CHAR, 0, 20, hc->icCommunicator()->communicator());
      }
   }
   assert(status == PV_SUCCESS);
   return status;
}

int dumpLoc(const PVLayerLoc * loc, int rank) {
   if( loc == NULL ) return PV_FAILURE;
   fprintf(stdout, "Rank %d: nx=%d, ny=%d, nf=%d, nxGlobal=%d, nyGlobal=%d, kx0=%d, ky0=%d\n",
          rank, loc->nx, loc->ny, loc->nf, loc->nxGlobal, loc->nyGlobal, loc->kx0, loc->ky0);
   return PV_SUCCESS;
}
#endif // PV_USE_MPI
