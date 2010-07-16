/*
 * geisler.cpp
 *
 */

#include <stdlib.h>
#include <time.h>

#include "../PetaVision/src/columns/HyPerCol.hpp"
#include "../PetaVision/src/connections/HyPerConn.hpp"
#include "../PetaVision/src/connections/KernelConn.hpp"
#include "../PetaVision/src/connections/CocircConn.hpp"
#include "../PetaVision/src/connections/GeislerConn.hpp"
#include "../PetaVision/src/connections/AvgConn.hpp"
#include "../PetaVision/src/layers/Movie.hpp"
#include "../PetaVision/src/layers/Image.hpp"
#include "../PetaVision/src/layers/Retina.hpp"
#include "../PetaVision/src/layers/V1.hpp"
#include "../PetaVision/src/layers/GeislerLayer.hpp"
#include "../PetaVision/src/io/ConnectionProbe.hpp"
#include "../PetaVision/src/io/GLDisplay.hpp"
#include "../PetaVision/src/io/PostConnProbe.hpp"
#include "../PetaVision/src/io/LinearActivityProbe.hpp"
#include "../PetaVision/src/io/PointProbe.hpp"
#include "../PetaVision/src/io/StatsProbe.hpp"

//#include "../PetaVisionsrc/io/imageio.hpp"


using namespace PV;

int main(int argc, char* argv[]) {

	//int iseed = time(NULL);
	//srand ( iseed );

	// create the managing hypercolumn
	//
	HyPerCol * hc = new HyPerCol("column", argc, argv);

	// create the visualization display
	//
	//GLDisplay * display = new GLDisplay(&argc, argv, hc, 2, 2);

#undef SPIKING
#ifdef SPIKING  // load geisler kernels from pvp file

	// create the image
	//
	const char * amoeba_filename = "../../MATLAB/amoeba/128_png/2/t/tar_0003_a.png";
	//display->setDelay(0);
	//display->setImage(image);

	// create the layers
	//
	HyPerLayer * image = new Image("Image", hc, amoeba_filename);
	HyPerLayer * retina = new Retina("Retina", hc);
	HyPerLayer * lgn = new V1("LGN", hc);
	HyPerLayer * lgninhff = new V1("LGNInhFF", hc);
	HyPerLayer * lgninh = new V1("LGNInh", hc);
	HyPerLayer * l1 = new V1("L1", hc);
	HyPerLayer * l1inhff = new V1("L1InhFF", hc);
	HyPerLayer * l1inh = new V1("L1Inh", hc);

	// create averaging layers
//	HyPerLayer * l1avg = new V1("L1Avg", hc);

	//display->addLayer(l1);
	//display->addLayer(l1inh);
	//display->addLayer(l1avg);


	// create the connections
	//

	HyPerConn * image_retina  =
		new KernelConn("Image to Retina",   hc, image, retina, CHANNEL_EXC);

	// retinal connections
	HyPerConn * r_lgn =
		new KernelConn("Retina to LGN", hc, retina, lgn,
			CHANNEL_EXC);
	HyPerConn * r_lgninhff =
		new KernelConn("Retina to LGNInhFF", hc, retina, lgninhff,
			CHANNEL_EXC);


	// LGN connections
//	HyPerConn * lgn_lgninhff =
//		new KernelConn("LGN to LGNInhFF", 	hc, lgn, lgninhff,
//			CHANNEL_EXC);
	HyPerConn * lgn_lgninh =
		new KernelConn("LGN to LGNInh", 	hc, lgn, lgninh,
			CHANNEL_EXC);
	HyPerConn * lgn_l1 =
		new KernelConn("LGN to L1",     	hc, lgn,  l1,
			CHANNEL_EXC);
	HyPerConn * lgn_l1inhff =
		new KernelConn("LGN to L1InhFF",    hc, lgn,  l1inhff,
			CHANNEL_EXC);


	// LGNInhFF connections
	HyPerConn * lgninhff_lgn =
		new KernelConn("LGNInhFF to LGN", 			hc, lgninhff, lgn,
			CHANNEL_INH);
//	HyPerConn * lgninhff_lgn_inhB =
//		new KernelConn("LGNInhFF to LGN InhB", 		hc, lgninhff, lgn,
//			CHANNEL_INHB);
//	HyPerConn * lgninhff_lgninhff_exc =
//		new KernelConn("LGNInhFF to LGNInhFF Exc", 	hc, lgninhff, lgninhff,
//			CHANNEL_EXC);
//	HyPerConn * lgninhff_lgninhff =
//		new KernelConn("LGNInhFF to LGNInhFF", 		hc, lgninhff, lgninhff,
//			CHANNEL_INH);
//	HyPerConn * lgninhff_lgninhff_inhB =
//		new KernelConn("LGNInhFF to LGNInhFF InhB", hc, lgninhff, lgninhff,
//			CHANNEL_INHB);
//	HyPerConn * lgninhff_lgninh =
//		new KernelConn("LGNInhFF to LGNInh", 		hc, lgninhff, lgninh,
//			CHANNEL_INH);
//	HyPerConn * lgninhff_lgninh_inhB =
//		new KernelConn("LGNInhFF to LGNInh InhB", 	hc, lgninhff, lgninh,
//			CHANNEL_INHB);


	// LGNInh connections
	HyPerConn * lgninh_lgn =
		new KernelConn("LGNInh to LGN", 		hc, lgninh, lgn,
			CHANNEL_INH);
	HyPerConn * lgninh_lgninh_exc =
		new KernelConn("LGNInh to LGNInh Exc", 	hc, lgninh, lgninh,
			CHANNEL_EXC);
	HyPerConn * lgninh_lgninh =
		new KernelConn("LGNInh to LGNInh", 		hc, lgninh, lgninh,
			CHANNEL_INH);


	// L1 connections
	const char * geisler_filename = "./input/test_amoeba10K_target_G1/4fc/geisler_clean.pvp";
	HyPerConn * l1_lgn =
		new KernelConn("L1 to LGN",  	hc, l1,     lgn,
			CHANNEL_EXC);
	HyPerConn * l1_lgninh =
		new KernelConn("L1 to LGNInh",  hc, l1,     lgninh,
			CHANNEL_EXC);
	HyPerConn * l1_l1 =
		new KernelConn("L1 to L1",      hc, l1,     l1,
			CHANNEL_EXC, geisler_filename);
//	HyPerConn * l1_l1inhff =
//		new CocircConn("L1 to L1InhFF", hc, l1,   	l1inhff,
//			CHANNEL_EXC);
	HyPerConn * l1_l1inh =
		new KernelConn("L1 to L1Inh",   hc, l1,     l1inh,
			CHANNEL_EXC, geisler_filename);


	// L1 Inh FF connections
	HyPerConn * l1inhff_l1 =
		new CocircConn("L1InhFF to L1",   			hc, l1inhff,  l1,
			CHANNEL_INH);
//	HyPerConn * l1inhff_l1_inhB =
//		new CocircConn("L1InhFF to L1 InhB",   		hc, l1inhff,  l1,
//			CHANNEL_INHB);
//	HyPerConn * l1inhff_l1inhff_exc =
//		new CocircConn("L1InhFF to L1InhFF Exc",   	hc, l1inhff,  l1inhff,
//			CHANNEL_EXC);
//	HyPerConn * l1inhff_l1inhff =
//		new CocircConn("L1InhFF to L1InhFF",   		hc, l1inhff,  l1inhff,
//			CHANNEL_INH);
//	HyPerConn * l1inhff_l1inhff_inhB =
//		new CocircConn("L1InhFF to L1InhFF InhB",   hc, l1inhff,  l1inhff,
//			CHANNEL_INHB);
//	HyPerConn * l1inhff_l1inh =
//		new CocircConn("L1InhFF to L1Inh",  		hc, l1inhff,  l1inh,
//			CHANNEL_INH);
//	HyPerConn * l1inhff_l1inh_inhB =
//		new CocircConn("L1InhFF to L1Inh InhB",   	hc, l1inhff,  l1inh,
//			CHANNEL_INHB);


	// L1 Inh connections
	HyPerConn * l1inh_l1 =
		new CocircConn("L1Inh to L1",   		hc, l1inh,  l1,
			CHANNEL_INH);
//	HyPerConn * l1inh_l1inhff =
//		new CocircConn("L1Inh to L1InhFF",  	hc, l1inh,  l1inhff,
//			CHANNEL_INH);
	HyPerConn * l1inh_l1inh_exc =
		new CocircConn("L1Inh to L1Inh Exc",   	hc, l1inh,  l1inh,
			CHANNEL_EXC);
	HyPerConn * l1inh_l1inh_inh =
		new CocircConn("L1Inh to L1Inh Inh",   	hc, l1inh,  l1inh,
			CHANNEL_INH);

	// create averaging connections
	//
//	HyPerConn * l1_l1avg   = new AvgConn("L1 to L1Avg", hc, l1, l1avg, CHANNEL_EXC, NULL);


	// add probes

	HyPerLayer * displayLayer = retina;

	const int nyDisplay = displayLayer->clayer->loc.ny;

#define DISPLAY2CONSOLE
#ifdef DISPLAY2CONSOLE

	LayerProbe * statsretina = new StatsProbe(BufActivity,     "Retina :");
	LayerProbe * statslgn = new StatsProbe(BufActivity,        "LGN :");
	LayerProbe * statslgninhff = new StatsProbe(BufActivity,     "LGNInhFF :");
	LayerProbe * statslgninh = new StatsProbe(BufActivity,     "LGNInh :");
	LayerProbe * statsl1 = new StatsProbe(BufActivity,         "L1     :");
	LayerProbe * statsl1inhff = new StatsProbe(BufActivity,       "L1InhFF:  ");
	LayerProbe * statsl1inh = new StatsProbe(BufActivity,       "L1Inh:  ");
//	LayerProbe * statsl1avg = new StatsProbe(BufActivity,      "L1Avg  :");

	retina->insertProbe(statsretina);
	lgn->insertProbe(statslgn);
	lgninhff->insertProbe(statslgninhff);
	lgninh->insertProbe(statslgninh);
	l1->insertProbe(statsl1);
	l1inhff->insertProbe(statsl1inhff);
	l1inh->insertProbe(statsl1inh);
//	l1avg->insertProbe(statsl1avg);

#endif

	int npad, nx, ny, nf;

	npad = lgn->clayer->loc.nPad;
	nx = lgn->clayer->loc.nx;
	ny = lgn->clayer->loc.ny;
	nf = lgn->clayer->loc.nBands;

#undef WRITE_VMEM
#ifdef WRITE_VMEM
	const char * Vmem_filename_LGNa1 = "Vmem_LGNa1.txt";
	LayerProbe * Vmem_probe_LGNa1 =
		new PointProbe(Vmem_filename_LGNa1, 51, 98, 0, "LGNA1:(51,98,0)");
	lgn->insertProbe(Vmem_probe_LGNa1);

	const char * Vmem_filename_LGNc1 = "Vmem_LGNc1.txt";
	LayerProbe * Vmem_probe_LGNc1 =
		new PointProbe(Vmem_filename_LGNc1, 80, 44, 0, "LGNC1:(80,44,0)");
	lgn->insertProbe(Vmem_probe_LGNc1);

	const char * Vmem_filename_LGNInhFFa1 = "Vmem_LGNInhFFa1.txt";
	LayerProbe * Vmem_probe_LGNInhFFa1 =
		new PointProbe(Vmem_filename_LGNInhFFa1, 51, 98, 0, "LGNInhA1:(51,98,0)");
	lgninhff->insertProbe(Vmem_probe_LGNInhFFa1);

	const char * Vmem_filename_LGNInhFFc1 = "Vmem_LGNInhFFc1.txt";
	LayerProbe * Vmem_probe_LGNInhFFc1 =
		new PointProbe(Vmem_filename_LGNInhFFc1, 80, 44, 0, "LGNInhFFC1:(80,44,0)");
	lgninhff->insertProbe(Vmem_probe_LGNInhFFc1);

	const char * Vmem_filename_LGNInha1 = "Vmem_LGNInha1.txt";
	LayerProbe * Vmem_probe_LGNInha1 =
		new PointProbe(Vmem_filename_LGNInha1, 51, 98, 0, "LGNInhA1:(51,98,0)");
	lgninh->insertProbe(Vmem_probe_LGNInha1);

	const char * Vmem_filename_LGNInhc1 = "Vmem_LGNInhc1.txt";
	LayerProbe * Vmem_probe_LGNInhc1 =
		new PointProbe(Vmem_filename_LGNInhc1, 80, 44, 0, "LGNInhC1:(80,44,0)");
	lgninh->insertProbe(Vmem_probe_LGNInhc1);

	const char * Vmem_filename_V1a1 = "Vmem_V1a1.txt";
	LayerProbe * Vmem_probe_V1a1 =
		new PointProbe(Vmem_filename_V1a1, 59, 91, 10, "V1A1:(59,91,10)");
	l1->insertProbe(Vmem_probe_V1a1);

	const char * Vmem_filename_V1c1 = "Vmem_V1c1.txt";
	LayerProbe * Vmem_probe_V1c1 =
		new PointProbe(Vmem_filename_V1c1, 80, 46, 5, "V1C1:(80,46,5)");
	l1->insertProbe(Vmem_probe_V1c1);

	const char * Vmem_filename_V1InhFFa1 = "Vmem_V1InhFFa1.txt";
	LayerProbe * Vmem_probe_V1InhFFa1 =
		new PointProbe(Vmem_filename_V1InhFFa1, 59, 91, 10, "V1InhFFA1:(59,91,10)");
	l1inhff->insertProbe(Vmem_probe_V1InhFFa1);

	const char * Vmem_filename_V1InhFFc1 = "Vmem_V1InhFFc1.txt";
	LayerProbe * Vmem_probe_V1InhFFc1 =
		new PointProbe(Vmem_filename_V1InhFFc1, 80, 46, 5, "V1InhFFC1:(80,46,5)");
	l1inh->insertProbe(Vmem_probe_V1InhFFc1);

	const char * Vmem_filename_V1Inha1 = "Vmem_V1Inha1.txt";
	LayerProbe * Vmem_probe_V1Inha1 =
		new PointProbe(Vmem_filename_V1Inha1, 59, 91, 10, "V1InhA1:(59,91,10)");
	l1inh->insertProbe(Vmem_probe_V1Inha1);

	const char * Vmem_filename_V1Inhc1 = "Vmem_V1Inhc1.txt";
	LayerProbe * Vmem_probe_V1Inhc1 =
		new PointProbe(Vmem_filename_V1Inhc1, 80, 46, 5, "V1InhC1:(80,46,5)");
	l1inh->insertProbe(Vmem_probe_V1Inhc1);
#endif

	if (0) { // ma
		LinearActivityProbe * laProbes[nyDisplay]; // array of ny pointers to PV::LinearActivityProbe

		for (int i = 0; i < nyDisplay; i++) {
			laProbes[i] = new PV::LinearActivityProbe(hc, PV::DimX, i, 0);
			retina->insertProbe(laProbes[i]);
		}
	}
	// run the simulation
	hc->initFinish();

	// write text weights
	const char * r_lgn_filename = "r_lgn_gauss.txt";
	HyPerLayer * pre = r_lgn->preSynapticLayer();
	npad = pre->clayer->loc.nPad;
	nx = pre->clayer->loc.nx;
	ny = pre->clayer->loc.ny;
	nf = pre->clayer->loc.nBands;
	r_lgn->writeTextWeights(r_lgn_filename, nf*(nx+npad)/2 + nf*(nx+2*npad)*(ny+2*npad)/2);

	const char * lgn_l1_filename = "lgn_l1_gauss.txt";
	pre = lgn_l1->preSynapticLayer();
	npad = pre->clayer->loc.nPad;
	nx = pre->clayer->loc.nx;
	ny = pre->clayer->loc.ny;
	nf = pre->clayer->loc.nBands;
	lgn_l1->writeTextWeights(lgn_l1_filename, nf*(nx+npad)/2 + nf*(nx+2*npad)*(ny+2*npad)/2);

	const char * lgninh_lgn_filename = "lgninh_lgn_gauss.txt";
	pre = lgninh_lgn->preSynapticLayer();
	npad = pre->clayer->loc.nPad;
	nx = pre->clayer->loc.nx;
	ny = pre->clayer->loc.ny;
	nf = pre->clayer->loc.nBands;
	lgninh_lgn->writeTextWeights(lgninh_lgn_filename, nf*(nx+npad)/2 + nf*(nx+2*npad)*(ny+2*npad)/2);

	const char * lgninh_lgninh_exc_filename = "lgninh_lgninh_gap.txt";
	pre = lgninh_lgninh_exc->preSynapticLayer();
	npad = pre->clayer->loc.nPad;
	nx = pre->clayer->loc.nx;
	ny = pre->clayer->loc.ny;
	nf = pre->clayer->loc.nBands;
	lgninh_lgninh_exc->writeTextWeights(lgninh_lgninh_exc_filename, nf*(nx+npad)/2 + nf*(nx+2*npad)*(ny+2*npad)/2);

//	const char * l1_lgn_filename = "l1_lgn_gauss.txt";
//	pre = l1_lgn->preSynapticLayer();
//	npad = pre->clayer->loc.nPad;
//	nx = pre->clayer->loc.nx;
//	ny = pre->clayer->loc.ny;
//	nf = pre->clayer->loc.nBands;
//	l1_lgn->writeTextWeights(l1_lgn_filename, nf*(nx+npad)/2 + nf*(nx+2*npad)*(ny+2*npad)/2);

//	const char * l1_lgninh_filename = "l1_lgninh_gauss.txt";
//	pre = l1_lgninh->preSynapticLayer();
//	npad = pre->clayer->loc.nPad;
//	nx = pre->clayer->loc.nx;
//	ny = pre->clayer->loc.ny;
//	nf = pre->clayer->loc.nBands;
//	l1_lgninh->writeTextWeights(l1_lgninh_filename, nf*(nx+npad)/2 + nf*(nx+2*npad)*(ny+2*npad)/2);

	const char * l1inhff_l1_filename = "l1inhff_l1_cocirc.txt";
	pre = l1inhff_l1->preSynapticLayer();
	npad = pre->clayer->loc.nPad;
	nx = pre->clayer->loc.nx;
	ny = pre->clayer->loc.ny;
	nf = pre->clayer->loc.nBands;
	l1inhff_l1->writeTextWeights(l1inhff_l1_filename, nf*(nx+npad)/2 + nf*(nx+2*npad)*(ny+2*npad)/2);

//	const char * l1_l1_filename = "l1_l1_geisler_exc.txt";
//	pre = l1_l1->preSynapticLayer();
//	npad = pre->clayer->loc.nPad;
//	nx = pre->clayer->loc.nx;
//	ny = pre->clayer->loc.ny;
//	nf = pre->clayer->loc.nBands;
//	l1_l1->writeTextWeights(l1_l1_filename, nf*(nx+npad)/2 + nf*(nx+2*npad)*(ny+2*npad)/2);
//	const char * l1_l1inh_filename = "l1_l1inh_geisler_exc.txt";
//	pre = l1_l1inh->preSynapticLayer();

//	npad = pre->clayer->loc.nPad;
//	nx = pre->clayer->loc.nx;
//	ny = pre->clayer->loc.ny;
//	nf = pre->clayer->loc.nBands;
//	l1_l1inh->writeTextWeights(l1_l1inh_filename, nf*(nx+npad)/2 + nf*(nx+2*npad)*(ny+2*npad)/2);
//	const char * l1inh_l1_filename = "l1inh_l1_geisler_inh.txt";
//	pre = l1inh_l1->preSynapticLayer();
//	npad = pre->clayer->loc.nPad;
//	nx = pre->clayer->loc.nx;
//	ny = pre->clayer->loc.ny;
//	nf = pre->clayer->loc.nBands;
//	l1inh_l1->writeTextWeights(l1inh_l1_filename, nf*(nx+npad)/2 + nf*(nx+2*npad)*(ny+2*npad)/2);

#else  // learn Geisler kernels

//	const char * amoeba_fileOfFileNames = "./input/128/test_amoeba10K_distractor_G1/4fc/fileNames.txt"; //
	const char * amoeba_fileOfFileNames = "./input/128/distractor40K_G3/fileNames.txt"; //
	float display_period = 1.0;
	Image * movie = new Movie("Movie", hc, amoeba_fileOfFileNames, display_period);
//	const char * amoeba_filename = "./input/test_amoebas/test0000.bmp"; // "./input/hard4.bmp"; //
//	Image * image = new Image("Image", hc, amoeba_filename);
	const char * image_file =  "./output/amoebaImage.tiff";
	movie->write(image_file);
	HyPerLayer * retina = new Retina("Retina", hc);
	LayerProbe * stats_retina = new StatsProbe(BufActivity,         "Retina :");
	retina->insertProbe(stats_retina);
	HyPerConn * image_retina  =
		new KernelConn("Movie to Retina",   hc, movie, retina, CHANNEL_EXC);
	HyPerLayer * l1 = new V1("L1", hc);
	HyPerConn * retina_l1 =
		new KernelConn("Retina to L1",   hc, retina,  l1,
			CHANNEL_EXC);
	HyPerConn * retina_l1_inh =
		new KernelConn("Retina to L1 Inh",   hc, retina,  l1,
			CHANNEL_INH);
	LayerProbe * statsl1 = new StatsProbe(BufActivity,         "L1     :");
	l1->insertProbe(statsl1);

#define TRAINING_TRIALS
#ifdef TRAINING_TRIALS

#define TRAINING_G2_TRIALS
#ifdef TRAINING_G2_TRIALS

	HyPerLayer * l1_geisler = new GeislerLayer("L1 Geisler", hc);
	LayerProbe * statsl1_geisler = new StatsProbe(BufActivity,         "L1 Geisler :");
	l1_geisler->insertProbe(statsl1_geisler);

	HyPerConn * l1_l1_geisler =
		new CocircConn("L1 to L1 Geisler",   			hc, l1,  	l1_geisler,
			CHANNEL_EXC);

	const char * geisler_filename_target = "./input/128/amoeba10K_G1/w3_last.pvp";
	HyPerConn * l1_l1_geisler_target =
		new KernelConn("L1 to L1 Geisler Target",   	hc, l1,    	l1_geisler,
			CHANNEL_INH, geisler_filename_target);
	const char * geisler_filename_distractor = "./input/128/distractor10K_G1/w3_last.pvp";
	HyPerConn * l1_l1_geisler_distractor =
		new KernelConn("L1 to L1 Geisler Distractor", 	hc, l1,     l1_geisler,
			CHANNEL_INH, geisler_filename_distractor);

#define TRAINING_G3_TRIALS
#ifdef TRAINING_G3_TRIALS

	HyPerLayer * l1_geisler2 = new GeislerLayer("L1 Geisler2", hc);
	LayerProbe * statsl1_geisler2 = new StatsProbe(BufActivity,         "L1 Geisler2 :");
	l1_geisler2->insertProbe(statsl1_geisler2);

	HyPerConn * l1_geisler_l1_geisler2 =
		new CocircConn("L1 Geisler to L1 Geisler2",   			hc, l1_geisler,  	l1_geisler2,
			CHANNEL_EXC);

	const char * geisler2_filename_target = "./input/128/amoeba10K_G2/w7_last.pvp";
	HyPerConn * l1_geisler_l1_geisler2_target =
		new KernelConn("L1 Geisler to L1 Geisler2 Target",   	hc, l1_geisler,    	l1_geisler2,
			CHANNEL_INH, geisler2_filename_target);
	const char * geisler2_filename_distractor = "./input/128/distractor10K_G2/w7_last.pvp";
	HyPerConn * l1_geisler_l1_geisler2_distractor =
		new KernelConn("L1 Geisler to L1 Geisler2 Distractor", 	hc, l1_geisler,     l1_geisler2,
			CHANNEL_INH, geisler2_filename_distractor);

#undef TRAINING_G4_TRIALS
#ifdef TRAINING_G4_TRIALS

	HyPerLayer * l1_geisler3 = new GeislerLayer("L1 Geisler3", hc);
	LayerProbe * statsl1_geisler3 = new StatsProbe(BufActivity,         "L1 Geisler3 :");
	l1_geisler3->insertProbe(statsl1_geisler3);

	HyPerConn * l1_geisler2_l1_geisler3 =
		new CocircConn("L1 Geisler2 to L1 Geisler3",   			hc, l1_geisler2,  	l1_geisler3,
			CHANNEL_EXC);

	const char * geisler3_filename_target = "./input/amoeba40K_G3/w10_last.pvp";
	HyPerConn * l1_geisler2_l1_geisler3_target =
		new KernelConn("L1 Geisler2 to L1 Geisler3 Target",   	hc, l1_geisler2,    	l1_geisler3,
			CHANNEL_INH, geisler3_filename_target);
	const char * geisler3_filename_distractor = "./input/distractor40K_G3/w10_last.pvp";
	HyPerConn * l1_geisler2_l1_geisler3_distractor =
		new KernelConn("L1 Geisler2 to L1 Geisler3 Distractor", 	hc, l1_geisler2,     l1_geisler3,
			CHANNEL_INH, geisler3_filename_distractor);

	HyPerLayer * l1_geisler4 = new V1("L1 Geisler4", hc);
	LayerProbe * statsl1_geisler4 = new StatsProbe(BufActivity,         "L1 Geisler4 :");
	l1_geisler4->insertProbe(statsl1_geisler4);

	HyPerConn * l1_geisler3_l1_geisler4 =
		new CocircConn("L1 Geisler3 to L1 Geisler4",   			hc, l1_geisler3,  	l1_geisler4,
			CHANNEL_EXC);

	HyPerConn * l1_geisler4_l1_geisler4 =
		new GeislerConn("L1 Geisler4 to L1 Geisler4",      hc, l1_geisler4,     l1_geisler4,
			CHANNEL_EXC);

#else  // !TRAINING_G4_TRIALS

	HyPerLayer * l1_geisler3 = new V1("L1 Geisler3", hc);
	LayerProbe * statsl1_geisler3 = new StatsProbe(BufActivity,         "L1 Geisler3 :");
	l1_geisler3->insertProbe(statsl1_geisler3);

	HyPerConn * l1_geisler2_l1_geisler3 =
		new CocircConn("L1 Geisler2 to L1 Geisler3",   			hc, l1_geisler2,  	l1_geisler3,
			CHANNEL_EXC);

	HyPerConn * l1_geisler3_l1_geisler3 =
		new GeislerConn("L1 Geisler3 to L1 Geisler3",      hc, l1_geisler3,     l1_geisler3,
			CHANNEL_EXC);

#endif // TRAINING_G4_TRIALS

#else  // ~TRAINING_G3_TRIALS

	HyPerLayer * l1_geisler2 = new V1("L1 Geisler2", hc);
	LayerProbe * statsl1_geisler2 = new StatsProbe(BufActivity,         "L1 Geisler2 :");
	l1_geisler2->insertProbe(statsl1_geisler2);

	HyPerConn * l1_geisler_l1_geisler2 =
		new CocircConn("L1 Geisler to L1 Geisler2",   			hc, l1_geisler,  	l1_geisler2,
			CHANNEL_EXC);

	HyPerConn * l1_geisler2_l1_geisler2 =
		new GeislerConn("L1 Geisler2 to L1 Geisler2",      hc, l1_geisler2,     l1_geisler2,
			CHANNEL_EXC);

#endif  // ~TRAINING_G3_TRIALS

#else  // ~TRAINING_G2_TRIALS

	HyPerConn * l1_l1 =
		new GeislerConn("L1 to L1",      hc, l1,     l1,
			CHANNEL_EXC);

#endif   // ~TRAINING_G2_TRIALS

#else  // ~TRAINING_TRIALS

	HyPerLayer * l1_geisler = new GeislerLayer("L1 Geisler", hc);
	HyPerConn * l1_l1_geisler =
		new CocircConn("L1 to L1 Geisler",   			hc, l1,  	l1_geisler,
			CHANNEL_EXC);
	const char * geisler_filename_target = "./input/128/amoeba10K_G1/w3_last.pvp";
	HyPerConn * l1_l1_geisler_target =
		new KernelConn("L1 to L1 Geisler Target",   	hc, l1,    	l1_geisler,
			CHANNEL_INH, geisler_filename_target);
	const char * geisler_filename_distractor = "./input/128/distractor10K_G1/w3_last.pvp";
	HyPerConn * l1_l1_geisler_distractor =
		new KernelConn("L1 to L1 Geisler Distractor", 	hc, l1,     l1_geisler,
			CHANNEL_INH, geisler_filename_distractor);
	LayerProbe * statsl1_geisler = new StatsProbe(BufActivity,         "L1 Geisler :");
	l1_geisler->insertProbe(statsl1_geisler);

	HyPerLayer * l1_geisler2 = new GeislerLayer("L1 Geisler2", hc);
	LayerProbe * statsl1_geisler2 = new StatsProbe(BufActivity,         "L1 Geisler2 :");
	l1_geisler2->insertProbe(statsl1_geisler2);

	HyPerConn * l1_geisler_l1_geisler2 =
		new CocircConn("L1 Geisler to L1 Geisler2",   			hc, l1_geisler,  	l1_geisler2,
			CHANNEL_EXC);
	const char * geisler2_filename_target = "./input/128/amoeba10K_G2/w7_last.pvp";
	HyPerConn * l1_geisler_l1_geisler2_target =
		new KernelConn("L1 Geisler to L1 Geisler2 Target",   	hc, l1_geisler,    	l1_geisler2,
			CHANNEL_INH, geisler2_filename_target);
	const char * geisler2_filename_distractor = "./input/128/distractor10K_G2/w7_last.pvp";
	HyPerConn * l1_geisler_l1_geisler2_distractor =
		new KernelConn("L1 Geisler to L1 Geisler2 Distractor", 	hc, l1_geisler,     l1_geisler2,
			CHANNEL_INH, geisler2_filename_distractor);

	HyPerLayer * l1_geisler3 = new GeislerLayer("L1 Geisler3", hc);
	LayerProbe * statsl1_geisler3 = new StatsProbe(BufActivity,         "L1 Geisler3 :");
	l1_geisler3->insertProbe(statsl1_geisler3);

	HyPerConn * l1_geisler2_l1_geisler3 =
		new CocircConn("L1 Geisler2 to L1 Geisler3",   			hc, l1_geisler2,  	l1_geisler3,
			CHANNEL_EXC);

	const char * geisler3_filename_target = "./input/128/amoeba40K_G3/w10_last.pvp";
	HyPerConn * l1_geisler2_l1_geisler3_target =
		new KernelConn("L1 Geisler2 to L1 Geisler3 Target",   	hc, l1_geisler2,    	l1_geisler3,
			CHANNEL_INH, geisler3_filename_target);
	const char * geisler3_filename_distractor = "./input/128/distractor40K_G3/w10_last.pvp";
	HyPerConn * l1_geisler2_l1_geisler3_distractor =
		new KernelConn("L1 Geisler2 to L1 Geisler3 Distractor", 	hc, l1_geisler2,     l1_geisler3,
			CHANNEL_INH, geisler3_filename_distractor);

	HyPerLayer * l1_geisler4 = new GeislerLayer("L1 Geisler4", hc);
	LayerProbe * statsl1_geisler4 = new StatsProbe(BufActivity,         "L1 Geisler4 :");
	l1_geisler4->insertProbe(statsl1_geisler4);

	HyPerConn * l1_geisler3_l1_geisler4 =
		new CocircConn("L1 Geisler3 to L1 Geisler4",   			hc, l1_geisler3,  	l1_geisler4,
			CHANNEL_EXC);

	const char * geisler4_filename_target = "./input/128/amoeba40K_G4/w13_last_129x129.pvp";
	HyPerConn * l1_geisler3_l1_geisler4_target =
		new KernelConn("L1 Geisler3 to L1 Geisler4 Target",   	hc, l1_geisler3,    	l1_geisler4,
			CHANNEL_INH, geisler3_filename_target);
	const char * geisler4_filename_distractor = "./input/128/distractor40K_G4/w13_last_129x129.pvp";
	HyPerConn * l1_geisler3_l1_geisler4_distractor =
		new KernelConn("L1 Geisler3 to L1 Geisler4 Distractor", 	hc, l1_geisler3,     l1_geisler4,
			CHANNEL_INH, geisler3_filename_distractor);

#endif

#endif

	hc->run();

	/* clean up (HyPerCol owns layers and connections, don't delete them) */
	delete hc;


	return 0;
}
