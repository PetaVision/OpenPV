/*
 * stdp.cpp
 *
 *  Created on: Apr 2, 2009
 *      Author: rasmussn
 */

#include <stdlib.h>

#include "Gratings.hpp"

#include <src/columns/HyPerCol.hpp>
#include <src/connections/RandomConn.hpp>
#include <src/layers/Image.hpp>
#include <src/layers/ImageCreator.hpp>
#include <src/layers/Movie.hpp>
#include <src/layers/Retina.hpp>
#include <src/layers/V1.hpp>
#include <src/io/ConnectionProbe.hpp>
#include <src/io/PostConnProbe.hpp>
#include <src/io/LinearActivityProbe.hpp>
#include <src/io/PointProbe.hpp>
#include <src/io/StatsProbe.hpp>

#include <src/io/imageio.hpp>

using namespace PV;

int main(int argc, char* argv[]) {
	// create the managing hypercolumn
	//
	HyPerCol * hc = new HyPerCol("column", argc, argv);

	// create the image
	//
	Image * image = new Gratings("Image", hc);

	// or create movie frames with the Movie class
	//
	if (0) {
		float displayPeriod = 20; // ms
		const char * files = "input/movies/movies.files";
		Movie * image = new Movie("Image", hc, files, displayPeriod);

		//
		// output first image frame
		image->write("frame0.tif");
	}
	// create the layers
	//
	HyPerLayer * retina = new Retina("Retina", hc, image);
	HyPerLayer * l1 = new V1("L1", hc);
	//HyPerLayer * l1Inh = new V1("L1Inh", hc);

	// create the connections
	//
	HyPerConn * r_l1 = new HyPerConn("Retina to L1", hc, retina, l1,
			CHANNEL_EXC);
	//HyPerConn * l1_inh = new HyPerConn("L1 to L1Inh", hc, l1, l1Inh,
	//		CHANNEL_EXC);
	//HyPerConn * inh_l1 = new HyPerConn("L1Inh to L1", hc, l1Inh, l1,
	//		CHANNEL_INH);

	// add probes

	HyPerLayer * displayLayer = retina;

	const int ny = displayLayer->clayer->loc.ny;
	PVLayerProbe * laProbes[ny]; // array of ny pointers to LinearActivityProbe

	for (int iy = 2; iy < ny - 2; iy++) {
		//      laProbes[iy] = new LinearActivityProbe(hc, DimX, iy, 0);
		//      displayLayer->insertProbe(laProbes[iy]);
	}

	PVLayerProbe * statsR = new StatsProbe(BufActivity, "R    :");
	PVLayerProbe * statsL1 = new StatsProbe(BufActivity, "L1   :");
	PVLayerProbe * statsL1Inh = new StatsProbe(BufActivity, "L1Inh:");

	//retina->insertProbe(statsR);
	//l1->insertProbe(statsL1);
	//l1Inh->insertProbe(statsL1Inh);

	PVLayerProbe * ptprobe0 = new PointProbe(16, 16, 0, "L1:(16,16)");
	PVLayerProbe * ptprobe1 = new PointProbe(17, 16, 0, "L1:(17,16)");
	PVLayerProbe * ptprobe2 = new PointProbe(18, 16, 0, "L1:(18,16)");
	PVLayerProbe * ptprobe3 = new PointProbe(19, 16, 0, "L1:(19,16)");
	PVLayerProbe * ptprobe4 = new PointProbe(20, 16, 0, "L1:(20,16)");
	//l1->insertProbe(ptprobe0);
	//   l1->insertProbe(ptprobe1);
	//   l1->insertProbe(ptprobe2);
	//   l1->insertProbe(ptprobe3);
	//   l1->insertProbe(ptprobe4);

	ConnectionProbe * cProbe0 = new PostConnProbe(16, 16, 0);
	ConnectionProbe * cProbe1 = new PostConnProbe(17, 16, 0);
	ConnectionProbe * cProbe2 = new PostConnProbe(18, 16, 0);
	ConnectionProbe * cProbe3 = new PostConnProbe(19, 16, 0);
	ConnectionProbe * cProbe4 = new PostConnProbe(20, 16, 0);
	//r_l1->insertProbe(cProbe0);
	//   r_l1->insertProbe(cProbe1);
	//   r_l1->insertProbe(cProbe2);
	//   r_l1->insertProbe(cProbe3);
	//   r_l1->insertProbe(cProbe4);

	if (1) { // ma
		LinearActivityProbe * laProbes[ny]; // array of ny pointers to PV::LinearActivityProbe

		for (unsigned int i = 0; i < ny; i++) {
			laProbes[i] = new PV::LinearActivityProbe(hc, PV::DimX, i, 0);
			retina->insertProbe(laProbes[i]);
			//l1->insertProbe(laProbes[i]);
			//l2->insertProbe(laProbes[i]);
			//l3->insertProbe(laProbes[i]);
			//l4->insertProbe(laProbes[i]);
			//l5->insertProbe(laProbes[i]);

		}
	}
	// run the simulation
	hc->initFinish();
	hc->run();

	/* clean up (HyPerCol owns layers and connections, don't delete them) */
	delete hc;

	return 0;
}
