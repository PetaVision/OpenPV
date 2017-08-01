#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * readheader.c, a C program for printing the header of a .pvp file
 * in human-readable form.
 * Usage: ./readheader file1 file2 file3 etc
 */

#define PARSE_SUCCESS 0
#define PARSE_FAILURE 1
#define PARSE_CANTOPEN 1
#define PARSE_NOHEADER 2
#define PARSE_BADHEADERSIZE 3
#define PARSE_BADNUMPARAMS 4
#define PARSE_BADFILETYPE 5
#define PARSE_NOMEM 12

#define INDEX_HEADER_SIZE 0
#define INDEX_NUM_PARAMS 1
#define INDEX_FILE_TYPE 2
#define INDEX_NX 3
#define INDEX_NY 4
#define INDEX_NF 5
#define INDEX_NUM_RECORDS 6
#define INDEX_RECORD_SIZE 7
#define INDEX_DATA_SIZE 8
#define INDEX_DATA_TYPE 9
#define INDEX_NX_PROCS 10
#define INDEX_NY_PROCS 11
#define INDEX_NX_EXTENDED 12
#define INDEX_NY_EXTENDED 13
#define INDEX_KX0 14
#define INDEX_KY0 15
#define INDEX_NBATCH 16
#define INDEX_NBANDS 17
#define INDEX_TIME 18

#define INDEX_WGT_NXP 0
#define INDEX_WGT_NYP 1
#define INDEX_WGT_NFP 2
#define INDEX_WGT_MIN 3
#define INDEX_WGT_MAX 4
#define INDEX_WGT_NUMPATCHES 5

// PVP_FILE_TYPE=1 removed Oct 21, 2016
#define PVP_ACT_FILE_TYPE 2
#define PVP_WGT_FILE_TYPE 3
#define PVP_NONSPIKING_ACT_FILE_TYPE 4
#define PVP_KERNEL_FILE_TYPE 5
#define PVP_ACT_SPARSEVALUES_FILE_TYPE 6

int parseheader(FILE *fid);

int main(int argc, char *argv[]) {
   int status = PARSE_SUCCESS;
   if (argc == 1 || (argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")))) {
      printf(
            "Usage: %s filename\nThis function parses the header of filename assuming filename is "
            "a pvp file.\n",
            argv[0]);
      return status;
   }
   int arg;
   for (arg = 1; arg < argc; arg++) {
      printf("%s\n", argv[arg]);
      FILE *fid = fopen(argv[arg], "r");
      if (fid == NULL) {
         fprintf(stderr, "Unable to open \"%s\": %s\n", argv[arg], strerror(errno));
         status = PARSE_CANTOPEN;
      }
      else {
         int status1 = parseheader(fid);
         if (status1 != PARSE_SUCCESS)
            status = PARSE_FAILURE;
         fclose(fid);
      }
   }
   status = status == PARSE_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
   return status;
}

int parseheader(FILE *fid) {
   int headersize;
   size_t numread = fread(&headersize, sizeof(int), 1, fid);
   if (numread != 1) {
      fprintf(stderr, "Unable to read headersize\n");
      return PARSE_NOHEADER;
   }
   int *params = (int *)calloc(headersize, 1);
   if (params == NULL) {
      fprintf(stderr, "Unable to allocate memory for header (header size=%d bytes)\n", headersize);
      return PARSE_NOMEM;
   }
   params[INDEX_HEADER_SIZE] = headersize;
   int numparams             = headersize / sizeof(int);
   numread                   = fread(&params[INDEX_NUM_PARAMS], sizeof(int), numparams - 1, fid);
   if (numread != numparams - 1) {
      fprintf(stderr, "Unable to read expected number of params (%d)\n", numparams);
      return PARSE_BADHEADERSIZE;
   }
   if (params[INDEX_NUM_PARAMS] * sizeof(int) != headersize) {
      fprintf(
            stderr,
            "Number of parameters (%d) inconsistent with headersize (%d)\n",
            params[INDEX_NUM_PARAMS],
            headersize);
      return PARSE_BADNUMPARAMS;
   }
   int correct_numparams = 0;
   int isweight          = 0;
   switch (params[INDEX_FILE_TYPE]) {
      case PVP_ACT_FILE_TYPE:
      case PVP_NONSPIKING_ACT_FILE_TYPE:
      case PVP_ACT_SPARSEVALUES_FILE_TYPE: correct_numparams = 20; break;
      case PVP_WGT_FILE_TYPE:
      case PVP_KERNEL_FILE_TYPE:
         correct_numparams = 26;
         isweight          = 1;
         break;
      default: correct_numparams = !params[1]; break;
   }
   if (correct_numparams != params[1]) {
      fprintf(
            stderr,
            "Number of parameters (%d) inconsistent with file type (%d)\n",
            params[INDEX_NUM_PARAMS],
            params[INDEX_FILE_TYPE]);
      return PARSE_BADFILETYPE;
   }
   printf("    Header size                        = %d\n", params[INDEX_HEADER_SIZE]);
   printf("    Number of params                   = %d\n", params[INDEX_NUM_PARAMS]);
   printf("    File type                          = %d\n", params[INDEX_FILE_TYPE]);
   printf("    nx                                 = %d\n", params[INDEX_NX]);
   printf("    ny                                 = %d\n", params[INDEX_NY]);
   printf("    nf                                 = %d\n", params[INDEX_NF]);
   printf("    Number of records                  = %d\n", params[INDEX_NUM_RECORDS]);
   printf("    Record size                        = %d\n", params[INDEX_RECORD_SIZE]);
   printf("    Data size                          = %d\n", params[INDEX_DATA_SIZE]);
   printf("    Data type                          = %d\n", params[INDEX_DATA_TYPE]);
   printf("    Number of processes in x-direction = %d\n", params[INDEX_NX_PROCS]);
   printf("    Number of processes in y-direction = %d\n", params[INDEX_NY_PROCS]);
   printf("    nx_Global                          = %d\n", params[INDEX_NX_EXTENDED]);
   printf("    ny_Global                          = %d\n", params[INDEX_NY_EXTENDED]);
   printf("    kx0                                = %d\n", params[INDEX_KX0]);
   printf("    ky0                                = %d\n", params[INDEX_KY0]);
   printf("    Batch width                        = %d\n", params[INDEX_NBATCH]);
   printf("    Number of bands                    = %d\n", params[INDEX_NBANDS]);
   double timestamp;
   memcpy(&timestamp, &params[INDEX_TIME], sizeof(double));
   printf("    Timestamp                          = %f\n", timestamp);

   if (isweight) {
      int *wgtparams = &params[20];
      printf("    Weight parameters:\n");
      printf("        nxp                            = %d\n", wgtparams[INDEX_WGT_NXP]);
      printf("        nyp                            = %d\n", wgtparams[INDEX_WGT_NYP]);
      printf("        nfp                            = %d\n", wgtparams[INDEX_WGT_NFP]);
      float minmax[2];
      memcpy(minmax, &wgtparams[INDEX_WGT_MIN], 2 * sizeof(float));
      printf("        min                            = %f\n", minmax[0]);
      printf("        max                            = %f\n", minmax[1]);
      printf("        Number of patches              = %d\n", wgtparams[INDEX_WGT_NUMPATCHES]);
   }
   free(params);
   return PARSE_SUCCESS;
}
