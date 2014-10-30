/*
 * pv.cpp
 *
 */

#include <columns/buildandrun.hpp>

int checkweights(HyPerCol * hc, int argc, char * argv[]);

int main(int argc, char * argv[]) {
   int paramfileabsent = pv_getopt_str(argc, argv, "-p", NULL);
   int num_cl_args;
   char ** cl_args;
   if( paramfileabsent ) {
      num_cl_args = argc + 2;
      cl_args = (char **) malloc((num_cl_args+1)*sizeof(char *));
      cl_args[0] = argv[0];
      cl_args[1] = strdup("-p");
      cl_args[2] = strdup("input/FourByFourGenerativeTest.params");
      for( int k=1; k<argc; k++) {
         cl_args[k+2] = strdup(argv[k]);
      }
      cl_args[num_cl_args] = NULL;
   }
   else {
      num_cl_args = argc;
      cl_args = argv;
   }
   int status = buildandrun(num_cl_args, cl_args, NULL, &checkweights)==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
   if( paramfileabsent ) {
      free(cl_args[1]);
      free(cl_args[2]);
      free(cl_args);
   }
   return status;
}

int checkweights(HyPerCol * hc, int argc, char * argv[]) {
   int weightsAidx = 2;
   int weightsBidx = 10;
   const char * weightsAname = "AnaRetina to Layer A";
   const char * weightsBname = "Arrow AnaLayer A to Layer B";
   BaseConnection * baseConn;
   baseConn = hc->getConnection(weightsAidx);
   HyPerConn * connA = dynamic_cast<HyPerConn *>(baseConn);
   if( strcmp(connA->getName(),weightsAname) ) {
      fprintf(stderr, "Expected connection %d to be named \"%s\"\n", weightsBidx, weightsAname);
      fprintf(stderr, "Instead it is named \"%s\"\n", connA->getName());
      exit(EXIT_FAILURE);
   }
   baseConn = hc->getConnection(weightsBidx);
   HyPerConn * connB = dynamic_cast<HyPerConn *>(baseConn);
   if( strcmp(connB->getName(),weightsBname) ) {
      fprintf(stderr, "Expected connection %d to be named \"%s\"\n", weightsBidx, weightsBname);
      fprintf(stderr, "Instead it is named \"%s\"\n", connB->getName());
      exit(EXIT_FAILURE);
   }
   assert(connA->xPatchSize() == 1);
   assert(connA->yPatchSize() == 1);
   assert(connA->fPatchSize() == 8);
   assert(connA->getNumDataPatches() == 16);
   assert(connB->xPatchSize() == 1);
   assert(connB->yPatchSize() == 1);
   assert(connB->fPatchSize() == 2);
   assert(connB->getNumDataPatches() == 8);

   pvwdata_t wgtA[8][16];
   pvwdata_t wgtB[2][8];

   for(int f=0; f<8; f++) {
      for(int k=0; k<16; k++) {
         wgtA[f][k] = connA->get_wData(0,k)[f]; // connA->getKernelPatch(0,k)->data[f];
      }
   }
   for(int f=0; f<2; f++) {
      for(int k=0; k<8; k++) {
         wgtB[f][k] = connB->get_wData(0,k)[f]; // connB->getKernelPatch(0,k)->data[f];
      }
   }
//
//   for(int f=0; f<8; f++) {
//      for(int k=0; k<16; k++) {
//        printf("%8.6f ", wgtA[f][k]);
//      }
//      printf("\n");
//   }
//   for( int k=0; k<72; k++ ) printf("-");
//   printf("\n");
//   for(int f=0; f<2; f++) {
//      for(int k=0; k<8; k++) {
//        printf("%8.6f ", wgtB[f][k]);
//      }
//      printf("\n");
//   }

   float nonbinaryA = -1;
   int worstk = 0;
   int worstf = 0;
   for( int f=0; f<8; f++ ) {
      for( int k=0; k<16; k++ ) {
         float z = fabs(fabs(wgtA[f][k]-0.5)-0.5);
         if( z>nonbinaryA ) {
            worstk = k;
            worstf = f;
            nonbinaryA = z;
         }
      }
   }
   printf("Largest discrepancy in weights A from zero or one is %f, when f=%d, k=%d\n",nonbinaryA, worstf, worstk);

   float nonbinaryB = -1;
   for( int f=0; f<2; f++ ) {
      for( int k=0; k<8; k++ ) {
         float z = fabs(fabs(wgtB[f][k]-0.5)-0.5);
         if( z>nonbinaryB ) {
            worstk = k;
            worstf = f;
            nonbinaryB = z;
         }
      }
   }
   printf("Largest discrepancy in weights B from zero or one is %f, when f=%d, k=%d\n",nonbinaryB, worstf, worstk);

   float nonbinarytol = 0.10;
   if( nonbinaryA > nonbinarytol || nonbinaryB > nonbinarytol ) {
      printf("Outside of tolerance %f.  FourbyFourLineTest failed.\n",nonbinarytol);
      exit(EXIT_FAILURE);
   }

   for( int f=0; f<8; f++ ) {
      for( int k=0; k<16; k++ ) {
         wgtA[f][k] = rint(wgtA[f][k]);
      }
   }

   for( int f=0; f<2; f++ ) {
      for( int k=0; k<8; k++ ) {
          wgtB[f][k] = rint(wgtB[f][k]);
      }
   }

   float correctA[8][16] = {
      {1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1},
      {1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0},
      {0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0},
      {0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0},
      {0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1}
   };

   float correctB[2][8] = {
      {1,1,1,1,0,0,0,0},
      {0,0,0,0,1,1,1,1}
   };

   int matches[8];
   int matchcount = 0;
   for( int f=0; f<8; f++ ) {
      matches[f] = -1;
      for( int g=0; g<8; g++ ) {
         int k;
         for( k=0; k<16; k++ ) {
            if( wgtA[f][k] != correctA[g][k] ) break;
         }
         if( k==16 ) {
            matchcount++;
            matches[f] = g;
         }
      }
   }
   bool faileded = false;
   for( int f=0; f<8; f++ ) {
      if( matches[f] >= 0 ) {
         printf("Row %d of wgtA matches row %d of correct answer\n", f, matches[f]);
      }
      else {
         printf("Row %d of wgtA did not match.\n", f);
         faileded = true;
      }
   }
   if( faileded ) {
      fprintf(stderr, "Not all rows of wgtA matched.\n");
      exit(EXIT_FAILURE);
   }
   assert(matchcount == 8);

   for( int k=0; k<8; k++ ) {
      int m = matches[k];
      printf("Column %d of wgtB should match column %d of correct answer...", k, m);
      int f;
      for( f=0; f<2; f++ ) {
         if( wgtB[f][k] != correctA[f][m] ) {
            break;
         }
      }
      if( f==2 ) {
         printf(".It does.\n");
      }
      else {
         printf(".Failure: wgtB(:,%d)=[%d; %d] but correctB(:,%d)=[%d; %d]\n", k, (int)wgtB[0][k], (int)wgtB[1][k], m, (int)correctB[0][m], (int)correctB[1][m]);
         faileded = true;
      }
   }
   if( faileded ) {
      fprintf(stderr, "Not all columns of wgtB matched.\n");
      exit(EXIT_FAILURE);
   }

   return PV_SUCCESS;
}
