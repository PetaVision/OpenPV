/*
 * LayerFunction.cpp
 *
 *  Created on: Nov 26, 2010
 *      Author: pschultz
 */

#include "LayerFunction.hpp"

namespace PV {

LayerFunction::LayerFunction(const char * name) {
	this->name = NULL;
    setName(name);
}

LayerFunction::~LayerFunction() {
    free(name);
}

void LayerFunction::setName(const char * name) {
    size_t len = strlen(name);
    if( this->name ) {
        free( this->name );
    }
    this->name = (char *) malloc( (len+1)*sizeof(char) );
    if( this->name) {
        strcpy(this->name, name);
    }
}

}  // end namespace PV
