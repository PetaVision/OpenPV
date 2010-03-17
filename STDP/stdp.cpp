/*
 * stdp.cpp
 *
 *  Created on: Apr 2, 2009
 *      Author: rasmussn
 */

#include <stdlib.h>

#include "RandomImage.hpp"

#include <src/columns/HyPerCol.hpp>
#include <src/connections/RandomConn.hpp>
#include <src/connections/AvgConn.hpp>
#include <src/layers/Gratings.hpp>
#include <src/layers/Bars.hpp>
#include <src/layers/Image.hpp>
#include <src/layers/ImageCreator.hpp>
#include <src/layers/Movie.hpp>
#include <src/layers/Retina.hpp>
#include <src/layers/V1.hpp>
#include <src/io/ConnectionProbe.hpp>
#include <src/io/GLDisplay.hpp>
#include <src/io/PostConnProbe.hpp>
#include <src/io/LinearActivityProbe.hpp>
#include <src/io/PointProbe.hpp>
#include <src/io/StatsProbe.hpp>

#include <src/io/imageio.hpp>

using namespace PV;

int main(int argc, char* argv[]) {

	int gratings_image = 0;
	int bars_image = 1;
	int random_image = 0;
	int movie_frames = 0;
	int point_probesI = 0;
	int point_probesII = 0;
	int conn_probes = 0;
	int postconn_probes = 0;
	int linact_probesX = 0; // activity along row
	int linact_probesY = 0; // activity along column
	int stat_probes = 0;
	int data_display = 0;

	// create the managing hypercolumn
	//
	HyPerCol * hc = new HyPerCol("column", argc, argv);

	// create the image

	Image * image;
	if (bars_image) {
		// bars Image
		Bars * bars = new Bars("Image", hc);
		image = bars;
	} else if (gratings_image) {
		// Gratings Image
		Gratings * gratings = new Gratings("Image", hc);
		image = gratings;
	} else if (random_image) {
		// Random Image
		image = new RandomImage("Image", hc);
	} else if (movie_frames) {
		// or create movie frames with the Movie class
		float displayPeriod = 20; // ms
		const char * files = "input/movies/movies.files";
		image = new Movie("Image", hc, files, displayPeriod);
		// output first image frame
		image->write("frame0.tif");
	} else {
		printf("choose an image type: abort\n");
		exit(-1);
	}

	// create the layers
	//
	HyPerLayer * retina = new Retina("Retina", hc, image);
	//HyPerLayer * av_retina = new V1("RetinaAvg", hc);
	HyPerLayer * l1 = new V1("L1", hc);
	//HyPerLayer * l1Inh = new V1("L1Inh", hc);

	if (data_display) {
		// create a run time data display
		int numRows = 2;
		int numCols = 2;
		GLDisplay * display = new GLDisplay(&argc, argv, hc, numRows, numCols);
		display->setDelay(50); // set wait delay of 50 ms each time step
		display->setImage(image);
		//display->addLayer(av_retina);
		display->addLayer(retina);
		display->addLayer(l1);
		//display->addLayer(l1Inh);
	}

	// create the connections
	//
	HyPerConn * r_l1 = new HyPerConn("Retina to L1", hc, retina, l1,
			CHANNEL_EXC);

	//HyPerConn * r_av = new AvgConn("Retina to RetinaAvg", hc, retina,
	//		av_retina, CHANNEL_EXC, NULL);

	//HyPerConn * l1_inh = new HyPerConn("L1 to L1Inh", hc, l1, l1Inh,
	//		CHANNEL_EXC);
	//HyPerConn * inh_l1 = new HyPerConn("L1Inh to L1", hc, l1Inh, l1,
	//		CHANNEL_INH);

	// add probes

	if (stat_probes) {
		LayerProbe * statsR = new StatsProbe(BufActivity, "R    :");
		LayerProbe * statsL1 = new StatsProbe(BufActivity, "L1   :");
		LayerProbe * statsL1Inh = new StatsProbe(BufActivity, "L1Inh:");

		retina->insertProbe(statsR);
		l1->insertProbe(statsL1);
		//l1Inh->insertProbe(statsL1Inh);
	}

	if (point_probesI) {
		LayerProbe * ptprobe0 = new PointProbe(16, 16, 0, "L1:(16,16)");
		LayerProbe * ptprobe1 = new PointProbe(17, 16, 0, "L1:(17,16)");
		LayerProbe * ptprobe2 = new PointProbe(18, 16, 0, "L1:(18,16)");
		LayerProbe * ptprobe3 = new PointProbe(19, 16, 0, "L1:(19,16)");
		LayerProbe * ptprobe4 = new PointProbe(20, 16, 0, "L1:(20,16)");
		l1->insertProbe(ptprobe0);
		l1->insertProbe(ptprobe1);
		l1->insertProbe(ptprobe2);
		l1->insertProbe(ptprobe3);
		l1->insertProbe(ptprobe4);
	}

	// insert point probes along a line
	if (point_probesII) {

		int locY = 4;
		int nx = l1->clayer->loc.nx;
		int ny = l1->clayer->loc.ny;
		PointProbe * ptProbes[nx]; // array of nx pointers to PointProbe

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

	if (conn_probes) {
		int n;
		int nx = retina->clayer->loc.nx;
		int ny = retina->clayer->loc.ny;
		//int nf = retina->clayer->loc.nBands;
		char filename[128] = { 0 };

		int numCProbes = nx * ny;

		ConnectionProbe * cProbe[numCProbes]; // array of pointers to ConnectionProbes
		n = 0;
		for (unsigned int iy = 0; iy < ny; iy++) {
			for (unsigned int ix = 0; ix < nx; ix++) {
				snprintf(filename, 127, "CP_%2d_%2d_%2d.dat", ix, iy, 0);
				cProbe[n] = new ConnectionProbe(filename,ix, iy, 0);
				//r_l1->insertProbe(cProbe1);
				cProbe[n]->setOutputIndices(true);
				cProbe[n]->setStdpVars(false);
				cProbe[n]->outputState(0.0, r_l1);
				cProbe[n]->setOutputIndices(false);
				//cProbe[n]->setStdpVars(true);
				n++;
			}
		}

	} // if(conn_probes)

	if (postconn_probes) {
		int n;
		int nx = l1->clayer->loc.nx;
		int ny = l1->clayer->loc.ny;
		int nf = l1->clayer->numFeatures;
		char filename[128] = { 0 };
		int numPCProbes = nx * ny * nf;

		PostConnProbe * pcProbe[numPCProbes];
		n = 0;
		for (unsigned int ik = 0; ik < nf; ik++) {
			for (unsigned int iy = 0; iy < ny; iy++) {
				for (unsigned int ix = 0; ix < nx; ix++) {
					snprintf(filename, 127, "PCP_%2d_%2d_%2d.dat", ix, iy, ik);
					pcProbe[n] = new PostConnProbe(filename,ix, iy, ik);
					//r_l1->insertProbe(pcProbe[n]);
					pcProbe[n]->setOutputIndices(true);
					pcProbe[n]->outputState(0.0, r_l1);
					pcProbe[n]->setOutputIndices(false);
					n++;
				}
			}
		}
	} // end if(postconn_probes)

	if (linact_probesX) { // ma

		HyPerLayer * displayLayer = retina;
		//HyPerLayer * displayLayer = l1;

		//const int marginWidth = displayLayer->clayer->loc.nPad;
		//const int ny = displayLayer->clayer->loc.ny + 2 * marginWidth;
		const int ny = displayLayer->clayer->loc.ny;
		//LinearActivityProbe * laProbes[2*marginWidth]; // array of ny pointers to LinearActivityProbe
		LinearActivityProbe * laProbes[ny];

		for (unsigned int iy = 0; iy < ny; iy++) {
			laProbes[iy] = new PV::LinearActivityProbe(hc, PV::DimX, iy, 0);
			displayLayer->insertProbe(laProbes[iy]);
		}

		//for (unsigned int iy = 0; iy < marginWidth ; iy++) {
		//			laProbes[iy] = new PV::LinearActivityProbe(hc, PV::DimX, iy, 0);
		//			displayLayer->insertProbe(laProbes[iy]);
		//		}

		//for (unsigned int iy = 0; iy < marginWidth ; iy++) {
		//			laProbes[marginWidth+iy] = new PV::LinearActivityProbe(hc, PV::DimX,
		//					marginWidth + ny +iy, 0);
		//			displayLayer->insertProbe(laProbes[marginWidth+iy]);
		//		}
	}

	if (linact_probesY) { // ma

		HyPerLayer * displayLayer = retina;
		//HyPerLayer * displayLayer = l1;

		const int nx = displayLayer->clayer->loc.nx;
		LinearActivityProbe * laProbes[nx]; // array of nx pointers to LinearActivityProbe

		for (unsigned int ix = 0; ix < nx; ix++) {
			laProbes[ix] = new PV::LinearActivityProbe(hc, PV::DimY, ix, 0);
			displayLayer->insertProbe(laProbes[ix]);
		}
	}

	// run the simulation
	hc->initFinish();
	hc->run();

	/* clean up (HyPerCol owns layers and connections, don't delete them) */
	delete hc;

	return 0;
}

