#include <string.h>

#include "readbin.h"

#define FREE(ptr)           \
    if (ptr != NULL)        \
    {                       \
        free(ptr);          \
        ptr = NULL;         \
    }                       \


void retrieve_args(int argc, char *argv[],
                   char **inputfile, char **outputfile);

int main_1(int argc, char *argv[])
{
    char *inputfile = NULL, *outputfile = NULL;
    FILE *inputfp = NULL, *outputfp = NULL;

    int params[7];

    retrieve_args(argc, argv, &inputfile, &outputfile);

    if ((inputfp = fopen(inputfile, "rb")) == NULL)
    {
        fprintf(stderr, "Error: Cannot open input binary file: %s\n",
                inputfile);
        exit(2);
    }

    if ((outputfile != NULL) && ((outputfp = fopen(outputfile, "w")) == NULL))
    {
        fprintf(stderr, "Error: Cannot open output file.\n");
        exit(2);
    }

    read_header(inputfp, params);
    write_header(outputfp, params);

    transfer_patch_data(inputfp, outputfp, params);

    fclose(inputfp);
    fclose(outputfp);

    FREE(inputfile);
    FREE(outputfile);

    return 0;
}


void retrieve_args(int argc, char *argv[],
                   char **inputfile, char **outputfile)
{
    unsigned int size = 0;

    if ((2 != argc) && (3 != argc))
    {
        fprintf(stderr, "Usage:\nreadbin SOURCE-BIN TARGET\n");
        fprintf(stderr, "or readbin SOURCE-BIN\n");
        exit(2);
    }

    if (2 == argc)
    {
        size = strlen(argv[1]);
        *inputfile = (char *) malloc(sizeof(char) * (size + 1));
        strncpy(*inputfile, argv[1], size);

        outputfile = NULL;
    }
    else if (3 == argc)
    {
        size = strlen(argv[1]);
        *inputfile = (char *) malloc(sizeof(char) * (size + 1));
        strncpy(*inputfile, argv[1], size);

        size = strlen(argv[2]);
        *outputfile = (char *) malloc(sizeof(char) * (size + 1));
        strncpy(*outputfile, argv[2], size);
    }
}

