/**
 * This file tests whether the HyPerCol parses
 * the -rows and -columns arguments correctly
 * Use with mpirun -np 6
 *
 */

#undef DEBUG_PRINT

#include "../src/include/pv_common.h"
#include "../src/columns/HyPerCol.hpp"
#include "../src/layers/ANNLayer.hpp"
#include "../src/io/io.h"
#include <assert.h>
#include <mpi.h>

int buildandverify(int argc, char * argv[]);
int verifyLoc(PV::HyPerCol * loc, int rows, int columns);
int dumpLoc(const PVLayerLoc * loc, int rank);

using namespace PV;

int main(int argc, char * argv[]) {
   int status;
#ifdef PV_USE_MPI
   int mpi_initialized_on_entry;
   MPI_Initialized(&mpi_initialized_on_entry);
   if( !mpi_initialized_on_entry ) MPI_Init(&argc, &argv);
   int numProcs;
   MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
#else
   int numProcs = 1;
#endif // PV_USE_MPI
   if( numProcs != 6) {
      fprintf(stderr, "%s: this test can only be used under MPI with exactly six processes.\n", argv[0]);
      // TODO Greater than six should be permissible, with the excess over 6 being idle
      exit(EXIT_FAILURE);
   }

#undef REQUIRE_RETURN // #define if the program should wait for carriage return before proceeding
#ifdef REQUIRE_RETURN
   int charhit;
   fflush(stdout);
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   if( rank == 0 ) {
      printf("Hit enter to begin! ");
      fflush(stdout);
      charhit = getc(stdin);
   }
#ifdef PV_USE_MPI
   int ierr;
   ierr = MPI_Bcast(&charhit, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif // PV_USE_MPI
#endif // REQUIRE_RETURN

#define TEST_MPI_SPECIFYROWCOLUMNS_ARGC 7
   char * cl_args[TEST_MPI_SPECIFYROWCOLUMNS_ARGC];
   cl_args[0] = argv[0];
   cl_args[1] = strdup("-p");
   cl_args[2] = strdup("input/test_mpi_specifyrowscolumns.params");
   cl_args[3] = strdup("-rows");
   cl_args[4] = strdup("2");
   cl_args[5] = strdup("-columns");
   cl_args[6] = strdup("3");
   buildandverify(TEST_MPI_SPECIFYROWCOLUMNS_ARGC, cl_args);

   free(cl_args[4]);
   cl_args[4] = strdup("3");
   free(cl_args[6]);
   cl_args[6] = strdup("2");
   buildandverify(TEST_MPI_SPECIFYROWCOLUMNS_ARGC, cl_args);

   for( int arg=1; arg<TEST_MPI_SPECIFYROWCOLUMNS_ARGC; arg++ ) {
      free(cl_args[arg]);
   }
#ifdef PV_USE_MPI
   if( !mpi_initialized_on_entry ) MPI_Finalize();
#endif PV_USE_MPI
   return status;
}

int buildandverify(int argc, char * argv[]) {
   for( int i=0; i<argc; i++ ) {
      assert(argv[i] != NULL);
   }
   PV::HyPerCol * hc = new PV::HyPerCol("column", argc, argv);
   PV::ANNLayer * layer = new PV::ANNLayer("layer", hc);
   PV::PVParams * params = hc->parameters();
   int rows = -1;
   int columns = -1;
   pv_getopt_int(argc, argv, "-rows", &rows);
   pv_getopt_int(argc, argv, "-columns", &columns);
   assert(rows >= 0 && columns >= 0);
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
                (loc->nb == params->value("layer", "marginWidth")) &&
                (loc->nxGlobal == nxGlobFromParams) &&
                (loc->nyGlobal == nyGlobFromParams) &&
                (loc->kx0 == loc->nx * (rank % columns)) &&
                (loc->ky0 == loc->ny * (rank / columns));

   PVLayerLoc mpiLoc;
   if( rank == 0 ) {
      printf("Testing with %d rows by %d columns of subprocesses.\n", rows, columns);
      if( testpassed ) {
         printf("Rank 0 passed.\n");
      }
      else {
         dumpLoc(loc, 0);
         fflush(stdout);
         fprintf(stderr, "Rank 0 FAILED\n");
         status = PV_FAILURE;
      }
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
   printf("Rank %d: nx=%d, ny=%d, nf=%d, nxGlobal=%d, nyGlobal=%d, kx0=%d, ky0=%d\n",
          rank, loc->nx, loc->ny, loc->nf, loc->nxGlobal, loc->nyGlobal, loc->kx0, loc->ky0);
   return PV_SUCCESS;
}
