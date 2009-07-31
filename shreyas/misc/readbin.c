
#include "readbin.h"

int read_header(FILE *inputfp, int *params)
{
    int num_params_to_read = 7;

    if ((num_params_to_read == 0) || (params == NULL))
    {
        fprintf(stderr, "Error; Invalid parameters to read header.\n");
        return 1;
    }

    if (num_params_to_read !=
        fread(params, sizeof(int), num_params_to_read, inputfp))
    {
        fprintf(stderr, "Error: Cannot read from input file.\n");
        return 1;
    }

    return 0;
}


int write_header(FILE *outputfp, int *params)
{
    if (params == NULL)
    {
        fprintf(stderr, "Error; Invalid parameters to read header.\n");
        return 1;
    }

    if (outputfp == NULL) outputfp = stdout;

    fprintf(outputfp, "numParams: %d \n nx: %d \n ny: %d \n nf: %d \n",
            params[0], params[1], params[2], params[3]);

    fprintf(outputfp, "Min weight value: %d \n Max weight value: %d \n numPatches: %d \n",
            params[4], params[5], params[6]);

    return 0;
}

int transfer_patch_data(FILE *inputfp, FILE *outputfp, int *params)
{
    int nf      = params[3];
    int min_val = params[4];
    int max_val = params[5];
    int num_patches = params[6];
    int i, err = 0;

    for (i = 0; i < num_patches; i++)
    {
        err = transfer_patch(inputfp, outputfp, nf, min_val, max_val);
        if (err != 0)
        {
            fprintf(stderr, "Error: Cannot read data for patch no: %d.\n", i);
            return 1;
        }
    }
    return 0;
}

int transfer_patch(FILE *inputfp, FILE *outputfp, int nf, int minVal, int maxVal)
{
    const int      bufSize = 4;
    int            i, ii, nItems;
    unsigned char  buf[bufSize];
    unsigned short nxny[2];
    float          patchdata;

    if ( fread(nxny, sizeof(unsigned short), 2, inputfp) != 2 ) return -1;

    nItems = (int) nxny[0] * (int) nxny[1] * (int) nf;

/*
    p->nx = (float) nxny[0];
    p->ny = (float) nxny[1];
    p->nf = nf;

    p->sf = 1;
    p->sx = nf;
    p->sy = (float) ( (int) p->nf * (int) p->nx );
*/
    i = 0;
    while (i < nItems)
    {
        if ( fread(buf, sizeof(unsigned char), bufSize, inputfp) != bufSize ) return -2;
        // data are packed into chars
        for (ii = 0; ii < bufSize; ii++)
        {
            patchdata = minVal + (maxVal - minVal) * ((float) buf[ii] / 255.0);
            if (++i >= nItems) break;
            fprintf(outputfp, "%f ", patchdata);
        }
    }

    fprintf(outputfp, "\n\n");
    return nItems;
}

