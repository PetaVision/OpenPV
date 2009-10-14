/*
 * pv_ca.cpp
 *
 *  Created on: Apr 2, 2009
 *      Author: rasmussn
 */

#include <stdlib.h>

//#include "Retina1D.hpp"
//#include "Retina.hpp"  - not needed as of Aug 28, 2009

#include "LinearPostConnProbe.hpp"

#include <src/columns/HyPerCol.hpp>
#include <src/io/ConnectionProbe.hpp>
#include <src/io/PostConnProbe.hpp>
#include <src/io/LinearActivityProbe.hpp>
#include <src/io/PointProbe.hpp>
#include <src/layers/ImageCreator.hpp>
#include <src/layers/Retina.hpp>
#include <src/layers/V1.hpp>
#include <src/connections/RandomConn.hpp>

#include <iostream>

using namespace PV;

int main(int argc, char* argv[]) {
	// create the Image class
	//Image * img = new Image();

	// create the managing hypercolumn
	HyPerCol * hc = new HyPerCol("column", argc, argv);

	int locY = 8;

	// create the layers
	ImageCreator * image = new ImageCreator("Image", hc);
	HyPerLayer * retina = new Retina("Retina", hc, image);

	HyPerLayer * l1 = new V1("L1", hc);
	//PV::HyPerLayer * l2     = new PV::V1("L2", hc);
	//PV::HyPerLayer * l3     = new PV::V1("L3", hc);
	//PV::HyPerLayer * l4     = new PV::V1("L4", hc);
	//PV::HyPerLayer * l5     = new PV::V1("L5", hc);

	//PV::HyPerConn * r_l1 =
	//		   new PV::RandomConn("Retina to L1", hc, retina, l1, CHANNEL_EXC);
	//   HyPerConn * r_l1 =
	//      new RandomConn("Retina to L1", hc, retina, l1, CHANNEL_EXC,PV::GAUSSIAN);
	HyPerConn * r_l1 = new HyPerConn("Retina to L1", hc, retina, l1,
			CHANNEL_EXC);

	//PV::HyPerConn * r_l1 =
	//      new PV::RandomConn("Retina to L1", hc, retina, l1, CHANNEL_EXC,PV::UNIFORM);

	//NOTE: The default is a uniform distribution in the wmin_init wmax_init range.
	//PV::HyPerConn * l1_l2 =
	// new PV::RandomConn("L1 to L2", hc, l1, l2, CHANNEL_EXC);

	//PV::HyPerConn * l2_l3 =
	// new PV::RandomConn("L2 to L3", hc, l2, l3, CHANNEL_EXC);

	//PV::HyPerConn * l3_l4 =
	//	   new PV::RandomConn("L3 to L4", hc, l3, l4, CHANNEL_EXC);

	//PV::HyPerConn * l4_l5 =
	//	   new PV::RandomConn("L4 to L5", hc, l4, l5, CHANNEL_EXC);

#if 0
	PV::PostConnProbe * pcProbe0 = new PV::LinearPostConnProbe(PV::DimX, locY, 0);
	PV::PostConnProbe * pcProbe1 = new PV::LinearPostConnProbe(PV::DimX, locY, 1);
	PV::PostConnProbe * pcProbe2 = new PV::LinearPostConnProbe(PV::DimX, locY, 1);
	PV::PostConnProbe * pcProbe3 = new PV::LinearPostConnProbe(PV::DimX, locY, 1);
	PV::PostConnProbe * pcProbe4 = new PV::LinearPostConnProbe(PV::DimX, locY, 1);

	r_l1->insertProbe(pcProbe0);
	l1_l2->insertProbe(pcProbe1);
	l2_l3->insertProbe(pcProbe2);
	l3_l4->insertProbe(pcProbe3);
	l4_l5->insertProbe(pcProbe4);
#endif

	PVParams* params = hc->parameters();

	int ny = retina->clayer->loc.ny;
	int nx = retina->clayer->loc.nx;

	// stdout probes
	//ConnectionProbe * cp_ul = new ConnectionProbe(5, 10, 0);
	//ConnectionProbe * cp_ur = new ConnectionProbe(10, 10, 0);

	// file probes
	//   ConnectionProbe * cp_ul = new ConnectionProbe("r5-10.probe", 5,10, 0);
	//   ConnectionProbe * cp_ul = new ConnectionProbe("r5-10.probe", 9,12, 0);
	//   ConnectionProbe * cp_ur = new ConnectionProbe("r9-10.probe",9, 10, 0);
	//   ConnectionProbe * cp_ll = new ConnectionProbe("r5-24.probe", 5,24, 0);
	//   ConnectionProbe * cp_lr = new ConnectionProbe("r9-24.probe",9,24, 0);

	//   r_l1->insertProbe(cp_ul);
	//   r_l1->insertProbe(cp_ur);
	//   r_l1->insertProbe(cp_ll);
	//   r_l1->insertProbe(cp_lr);

	LinearActivityProbe * laProbes[ny]; // array of ny pointers to PV::LinearActivityProbe

	for (unsigned int i = 0; i < ny; i++) {
		laProbes[i] = new PV::LinearActivityProbe(hc, PV::DimX, i, 0);
		//retina->insertProbe(laProbes[i]);
		l1->insertProbe(laProbes[i]);
		//l2->insertProbe(laProbes[i]);
		//l3->insertProbe(laProbes[i]);
		//l4->insertProbe(laProbes[i]);
		//l5->insertProbe(laProbes[i]);

	}

	if (0) {
		ConnectionProbe * cp_ul = new ConnectionProbe(5, 10, 0);
		r_l1->insertProbe(cp_ul);

		ConnectionProbe * cp_ur = new ConnectionProbe(14, 10, 0);
		r_l1->insertProbe(cp_ur);

		PointProbe* ptprobeR1 = new PointProbe(5, 10, 0, "R1 :");
		PointProbe* ptprobe1 = new PointProbe(5, 10, 0, "ul:");
		retina->insertProbe(ptprobeR1);
		l1->insertProbe(ptprobe1);

		PointProbe* ptprobeR2 = new PointProbe(14, 10, 0, "R2 :");
		PointProbe* ptprobe2 = new PointProbe(14, 10, 0, "ur:");
		retina->insertProbe(ptprobeR2);
		l1->insertProbe(ptprobe2);
	}

	// insert point probes along a line
	if (0) {
		PointProbe * ptProbes[nx]; // array of nx pointers to PointProbe
		locY = 10;

		for (unsigned int i = 0; i < nx; i++) {
			char str[10];
			if (i < 10)
				sprintf(str, " %d: ", i);
			else
				sprintf(str, "%d: ", i);
			ptProbes[i] = new PointProbe(i, locY, 0, str);
			//retina->insertProbe(rProbes[i]);
			l1->insertProbe(ptProbes[i]);
		}
	}

	// run the simulation
	hc->initFinish();
	hc->run();

	/* clean up (HyPerCol owns layers and connections, don't delete them) */
	delete hc;

	return 0;
}
