/*
 * CreateMovies.cpp
 *
 *  Created on: Dec. 12, 2009
 *      Author: Wentao
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../columns/InterColComm.hpp"
#include "../include/LayerLoc.h"

#include "CreateMovies.hpp"

#define Pi 3.14159265

#define DEBUG_OUTPUTIMAGES
#ifdef DEBUG_OUTPUTIMAGES
	int numFrame = 0;
	int MaxNumberFrames = 20;
#endif


CreateMovies_Params DefaultCMParams={
	32,//nx
	32,//ny
	255,//foregroundval
	0,//backgroudval
	1,//isgray
	0,//rotateangle
	0,//centerx
	0,//centery
	8,//period
	3,//linewidth
	2,//vx
	2,//vy
	5,//vr
	-1,//isshiftx
	0,//isshifty
	0,//isrotate
};

namespace PV {

CreateMovies::CreateMovies(const char * name, HyPerCol * hc)
	:Image(name, hc)
{
	initialize_Movies(hc);
}

CreateMovies::~CreateMovies() {
	if (CMParams != NULL) free(CMParams);
}

int CreateMovies::initialize_Movies(HyPerCol * hc){

	CMParams = NULL;
	PVParams * pvparams = hc->parameters();
	displayPeriod = pvparams->value(name, "displayPeriod", 20.0);
	lastDisplayTime = hc->simulationTime();
	nextDisplayTime = hc->simulationTime() + displayPeriod;

	flagx = 1;
	flagy = 1;
	flagr = 1;

	setParams(pvparams, &DefaultCMParams);
	CreateMovies_Params * cp = (CreateMovies_Params *) CMParams;

	//LayerLoc imageLoc = hc->getImageLoc();
	loc.nx = cp->nx;
	loc.ny = cp->ny;
	loc.nBands = 1;
	loc.nPad = pvparams->value(name, "marginWidth", 0);

	if (data != NULL) free(data);
	size_t dn = loc.nBands * (loc.nx + 2*loc.nPad)* (loc.ny + 2*loc.nPad)* sizeof(pvdata_t);
	data = (pvdata_t *) malloc(dn);
	assert(data != NULL);
	memset((pvdata_t *)data, cp->backgroudval, dn);
	Transform(0, 0, 0);

#ifdef DEBUG_OUTPUTIMAGES
	int T = hc->simulationTime() ;
	char title[1000];
	::sprintf(title,"output/images/%05d.tif",T);
	write(title);
#endif
	return 0;
}


int CreateMovies::setParams(PVParams * params, CreateMovies_Params * p)
{

   CMParams = (CreateMovies_Params *) malloc(sizeof(*p));
   assert(CMParams != NULL);
   memcpy(CMParams, p, sizeof(CreateMovies_Params));

   CreateMovies_Params * cp = CMParams;

   if (params->present(name, "nx"))  			cp->nx  = params->value(name, "nx");
   if (params->present(name, "ny"))  			cp->ny = params->value(name, "ny");
   if (params->present(name, "foregroundval"))  cp->foregroundval = params->value(name, "foregroundval");
   if (params->present(name, "backgroudval"))   cp->backgroudval = params->value(name, "backgroudval");
   if (params->present(name, "isgray"))    		cp->isgray  = params->value(name, "isgray");
   if (params->present(name, "rotateangle"))    cp->rotateangle  = params->value(name, "rotateangle");
   if (params->present(name, "centerx"))      	cp->centerx = params->value(name, "centerx");
   if (params->present(name, "centery"))  		cp->centery  = params->value(name, "centery");
   if (params->present(name, "period")) 		cp->period  = params->value(name, "period");
   if (params->present(name, "linewidth")) 		cp->linewidth  = params->value(name, "linewidth");
   if (params->present(name, "vx")) 			cp->vx  = params->value(name, "vx");
   if (params->present(name, "vy")) 			cp->vy  = params->value(name, "vy");
   if (params->present(name, "vr")) 			cp->vr  = params->value(name, "vr");
   if (params->present(name, "isshiftx")) 		cp->isshiftx  = params->value(name, "isshiftx");
   if (params->present(name, "isshifty")) 		cp->isshifty  = params->value(name, "isshifty");
   if (params->present(name, "isrotate")) 		cp->isrotate  = params->value(name, "isrotate");

   return 0;
}


int CreateMovies::Rotate(const float DAngle, const int centerx, const int centery){

		CreateMovies_Params *param = (CreateMovies_Params *)CMParams;

		int Nx = loc.nx + 2*loc.nPad;
		int Ny = loc.ny + 2*loc.nPad;

		param->rotateangle+=DAngle;
		if (param->rotateangle>=180){
			param->rotateangle=((int)(param->rotateangle)) % 180 + param->rotateangle-(int)param->rotateangle;
		}

		if (param->rotateangle<=-180){
			param->rotateangle=((int)(param->rotateangle)) % 180 + param->rotateangle-(int)param->rotateangle;
		}

		float Agl = Pi*param->rotateangle/180.0;
		int cx = Nx/2.0, cy = Ny/2.0;
		int period = param->period;
		int linewidth = param->linewidth;
		pvdata_t foregroundval = param->foregroundval;
		pvdata_t backgroudval  = param->backgroudval;

		size_t dn=loc.nBands * Nx * Ny * sizeof(pvdata_t);
		if (data == NULL){
			data = (pvdata_t *) malloc(dn);
		}
		assert(data != NULL);

		memset((pvdata_t *)data, backgroudval, dn);


		float cs = ::cos((double)Agl),  sn = ::sin((double)Agl);
		int i,j,i1,j1,w,n,m;



		float a = centerx*cs + centery*sn;
		//float b = centery*cs - centerx*sn;

		int cx1 = Nx-cx;
		int cy1 = Ny-cy;
		float Pd = 2*Pi/period;
		float fb = (foregroundval-backgroudval)/2;

		for(i=-cx; i<cx1; i++){
			for(j=-cy; j<cy1; j++){
				i1=(int)(i*cs + j*sn-a);//(int)(i*cs+0.5 + j*sn-a + 0.5);
				//j1=j*cs - i*sn-b;
				if (param->isgray){
					data[i+cx+(j+cy)*Nx] = backgroudval + (1+::cos(Pd*i1))*fb;
				}
				else{
					int t=i1%period;
					if ((t>=0 && t< linewidth)||(t<0 && t+period < linewidth)){
						data[i+cx+(j+cy)*Nx] = foregroundval;
					}
				}

			}
		}

		return 0;
}

int CreateMovies::Transform(const float DAngle,const int Dx,const int Dy){

		CreateMovies_Params *param =  CMParams;

		int Nx = loc.nx + 2*loc.nPad;
		int Ny = loc.ny + 2*loc.nPad;

		param->centerx +=  Dx;
		param->centerx  =  (int)param->centerx % (int)(10*Nx);
		param->centery +=  Dy;
		param->centery  =  (int)param->centery % (int)(10*Ny);

		return Rotate(DAngle, param->centerx, param->centery);
}

bool CreateMovies::updateImage(float time, float dt){


	if(time - lastDisplayTime < dt/2) {
		return true;
	}

	if (time < nextDisplayTime) {
		      return false;
	}

	lastDisplayTime = time;

	nextDisplayTime += displayPeriod;

	CreateImages();

#ifdef DEBUG_OUTPUTIMAGES
	if (numFrame < MaxNumberFrames){
		int T = nextDisplayTime - displayPeriod;
		char title[1000];
		::sprintf(title,"output/images/%05d.tif",T);
		write(title);
		numFrame++;
	}
#endif

	return true;
}

bool CreateMovies::CreateImages(){

	CreateMovies_Params *param = (CreateMovies_Params *)CMParams;

	int isshiftx = param->isshiftx;
	int isshifty = param->isshifty;
	int isrotate = param->isrotate;

	/*Transform(param->vr,param->vx,param->vy);*/
	// Create different pattern image sequences
	while(1){
	int flag = 0;
	if (isshiftx != 0){
		if (isshiftx<0)
			Transform(0,param->vx,0);
		else{
			flag = flagx % (int)isshiftx;
			if (flag != 0){
				Transform(0,param->vx,0);
				flagy = 0;
				flagr = 0;
				flagx++;
			}
			else{
				flagx = 0;
				flagy = 1;
				flagr = 1;
			}
			break;
		}
	}

	if (isshifty != 0){
		if (isshifty<0)
			Transform(0,0,param->vy);
		else{
			flag = flagy % (int)isshifty;
			if (flag != 0){
				Transform(0,0,param->vy);
				flagx = 0;
				flagr = 0;
				flagy++;
			}
			else{
				flagx = 1;
				flagy = 0;
				flagr = 1;
			}
			break;
		}

	}

	if (isrotate != 0){
		if (isrotate<0)
			Transform(param->vr,0,0);
		else{
			flag = flagr % (int)isrotate;
			if (flag != 0){
				Transform(param->vr,0,0);
				flagx = 0;
				flagy = 0;
				flagr++;
			}
			else{
				flagx = 1;
				flagy = 1;
				flagr = 0;
			}
			break;
		}
	}

	break;
	}//while
	return true;
}

}//namespace PV





