/*
 * ColProbe.cpp
 *
 *  Created on: Nov 25, 2010
 *      Author: pschultz
 */

#include "ColProbe.hpp"

namespace PV {

ColProbe::ColProbe() {
    fp = stdout;
}

ColProbe::ColProbe(const char * filename, HyPerCol * hc) {
    char * path;
    const char * output_path = hc->getOutputPath();
    size_t len = strlen(output_path) + strlen(filename) + 1;
    path = (char *) malloc( len * sizeof(char) );
    sprintf(path, "%s/%s", output_path, filename);
    fp = fopen(path, "w");
    free(path);
}

ColProbe::~ColProbe() {
    if( fp != NULL && fp != stdout) {
        fclose(fp);
    }
}

}  // end namespace PV
