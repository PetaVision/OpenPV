/*
 * KernelConnProbe.cpp
 *
 *  Created on: Jan 10, 2011
 *      Author: pschultz
 */

#include "KernelConnDumpProbe.hpp"

namespace PV {

KernelConnDumpProbe::KernelConnDumpProbe() : ConnectionProbe(0) {
    textNotBinaryFlag = true;
    stdoutNotFileFlag = true;
    filenameFormatString = NULL;

    initializeProbe();
}

KernelConnDumpProbe::KernelConnDumpProbe(const char * filenameformatstr) : ConnectionProbe(0) {
    textNotBinaryFlag = false;
    stdoutNotFileFlag = false;
    filenameFormatString = (char *) malloc( ( strlen(filenameformatstr)+1 )*sizeof(char) );
    strcpy(filenameFormatString, filenameformatstr);

    initializeProbe();
}

KernelConnDumpProbe::KernelConnDumpProbe(const char * filenameformatstr, bool textNotBinary) : ConnectionProbe(0) {
    textNotBinaryFlag = textNotBinary;
    stdoutNotFileFlag = false;
    filenameFormatString = (char *) malloc( ( strlen(filenameformatstr)+1 )*sizeof(char) );
    strcpy(filenameFormatString, filenameformatstr);

    initializeProbe();
}

KernelConnDumpProbe::~KernelConnDumpProbe() {
    if( !stdoutNotFileFlag )
        free( filenameFormatString );
}

int KernelConnDumpProbe::outputState(float time, HyPerConn * c) {
    FILE * fp;
    if( stdoutNotFileFlag ) {
        fp = stdout;
    }
    else {
        int len = printf(filenameFormatString,time);
        printf("\n");
        if( len <= 0 ) return EXIT_FAILURE;
        char * filename = (char *) malloc( (len+1)*sizeof(char) );
        if( !filename ) return EXIT_FAILURE;
        sprintf(filename, filenameFormatString, time);
        fp = fopen(filename,"w");
        if( !filename ){ free(filename); return EXIT_FAILURE; }
    }
	int nfp = c->fPatchSize();
	int nxp = c->xPatchSize();
	int nyp = c->yPatchSize();
    int numPatches = c->numDataPatches(0);
    if( textNotBinaryFlag ) {
        fprintf(fp, "time = %f\n", time);
        fprintf(fp, "nfp=%d, nxp=%d, nyp=%d, numPatches=%d\n",nfp,nxp,nyp,numPatches);
        for( int k=0; k<numPatches; k++) {
            fprintf(fp, "Patch %d\n", k);
            PVPatch * patch = ((KernelConn *) c)->getKernelPatch(k);
            for( int y=0; y<nyp; y++ ) {
                for( int x=0; x<nxp; x++ ) {
                    fprintf(fp, "    x=%d, y=%d\n",x,y);
                    for( int f=0; f<nfp; f++ ) {
                        int idx = kIndex(x,y,f,nxp,nyp,nfp);
                        pvdata_t value = patch->data[idx];
                        fprintf(fp,"        feature %d: value=%g\n",f,value);
                    }
                }
            }
        }
    }
    else {
        fwrite(&time,sizeof(time), (size_t) 1, fp);
        fwrite(&nfp,sizeof(nfp), (size_t) 1, fp);
        fwrite(&nxp,sizeof(nxp), (size_t) 1, fp);
        fwrite(&nyp,sizeof(nyp), (size_t) 1, fp);
        fwrite(&numPatches,sizeof(numPatches), (size_t) 1, fp);
        for( int k=0; k<numPatches; k++) {
        	PVPatch * patch = ((KernelConn *) c)->getKernelPatch(k);
            fwrite(patch->data, sizeof(pvdata_t), (size_t) (nfp*nxp*nyp), fp);
        }
    }
    if( !stdoutNotFileFlag ) {
        fclose(fp);
    }

    return EXIT_SUCCESS;
}

int KernelConnDumpProbe::validateFormatString(const char * formatstr) {
    return EXIT_SUCCESS;
}

int KernelConnDumpProbe::initializeProbe() {
    return validateFormatString(filenameFormatString);
}

}  // end namespace PV
