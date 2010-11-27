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

ColProbe::ColProbe(const char * filename) {
    char * path;
    size_t len = strlen(OUTPUT_PATH) + strlen(filename) + 1;
    path = (char *) malloc( len * sizeof(char) );
    sprintf(path, "%s%s", OUTPUT_PATH, filename);
    fp = fopen(path, "w");
    free(path);
}

ColProbe::~ColProbe() {
    if( fp != NULL && fp != stdout) {
        fclose(fp);
    }
}

}  // end namespace PV
