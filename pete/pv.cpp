/*
 * pv.cpp
 *
 */

#include <stdlib.h>
#include <time.h>

#include "../PetaVision/src/columns/HyPerCol.hpp"
#include "../PetaVision/src/connections/HyPerConn.hpp"
#include "../PetaVision/src/connections/KernelConn.hpp"
#include "../PetaVision/src/connections/GeislerConn.hpp"
#include "../PetaVision/src/connections/CocircConn.hpp"
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

// #include "GenLatConn.hpp"

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

        PVParams * params = hc->parameters();

        // Is this run being used to train?
        float training_flag = params->value("column","training_flag");
        const char * fileOfFileNames = params->getFilename("ImageFileList");
        // const char * outputDir = params->getFilename("OutputDir");

        float display_period = 1.0;
        Image * movie = new Movie("Movie", hc, fileOfFileNames, display_period);
        const char * image_file =  "./output/outImage.tiff";
        movie->write(image_file);
        HyPerLayer * retina = new Retina("Retina", hc);
        LayerProbe * stats_retina = new StatsProbe(BufActivity,   "Retina :");
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
        LayerProbe * statsl1 = new StatsProbe(BufActivity,  "L1     :");
        l1->insertProbe(statsl1);

        if( training_flag ) {
            HyPerConn * l1_l1 =
                    new GeislerConn("L1 to L1", hc, l1, l1, CHANNEL_EXC);
        }
        else {
			HyPerLayer * l1_geisler = new GeislerLayer("L1 Geisler", hc);
			HyPerConn * l1_l1_geisler =
				new CocircConn("L1 to L1 Geisler", hc, l1, l1_geisler,
					CHANNEL_EXC);
			const char * geisler_filename_target = params->getFilename("TargetWgts");
			HyPerConn * l1_l1_geisler_target =
				new KernelConn("L1 to L1 Geisler Target", hc, l1,  l1_geisler,
					CHANNEL_INH, geisler_filename_target);
			const char * geisler_filename_distractor = params->getFilename("DistractorWgts");
			HyPerConn * l1_l1_geisler_distractor =
				new KernelConn("L1 to L1 Geisler Distractor", 	hc, l1, l1_geisler,
					CHANNEL_INH, geisler_filename_distractor);
			LayerProbe * statsl1_geisler = new StatsProbe(BufActivity,  "L1 Geisler :");
			l1_geisler->insertProbe(statsl1_geisler);

			HyPerLayer * l2_geisler = new GeislerLayer("L2 Geisler", hc);
			LayerProbe * statsl2_geisler = new StatsProbe(BufActivity,  "L2 Geisler :");
			l2_geisler->insertProbe(statsl2_geisler);

			HyPerConn * l1_geisler_l2_geisler =
				new CocircConn("L1 Geisler to L2 Geisler", hc, l1_geisler, l2_geisler,
					CHANNEL_EXC);
			//const char * geisler2_filename_target = "./input/256/target40K_G2/w7_last.pvp";
			HyPerConn * l1_geisler_l2_geisler_target =
				new KernelConn("L1 Geisler to L2 Geisler Target", hc, l1_geisler, l2_geisler,
					CHANNEL_INH, geisler_filename_target);
			//const char * geisler2_filename_distractor = "./input/256/distractor40K_G2/w7_last.pvp";
			HyPerConn * l1_geisler_l2_geisler_distractor =
				new KernelConn("L1 Geisler to L2 Geisler Distractor", hc, l1_geisler, l2_geisler,
					CHANNEL_INH, geisler_filename_distractor);

			HyPerLayer * l3_geisler = new GeislerLayer("L3 Geisler", hc);
			LayerProbe * statsl3_geisler = new StatsProbe(BufActivity, "L3 Geisler :");
			l3_geisler->insertProbe(statsl3_geisler);

			HyPerConn * l2_geisler_l3_geisler =
				new CocircConn("L2 Geisler to L3 Geisler", hc, l2_geisler, l3_geisler,
					CHANNEL_EXC);

			//const char * geisler3_filename_target = "./input/256/amoeba20K_97x97_G3/w10_last.pvp";
			HyPerConn * l2_geisler_l3_geisler_target =
				new KernelConn("L2 Geisler to L3 Geisler Target", hc, l2_geisler, l3_geisler,
					CHANNEL_INH, geisler_filename_target);
			//const char * geisler3_filename_distractor = "./input/256/distractor20K_97x97_G3/w10_last.pvp";
			HyPerConn * l2_geisler_l3_geisler_distractor =
				new KernelConn("L2 Geisler to L3 Geisler Distractor", 	hc, l2_geisler,   l3_geisler,
					CHANNEL_INH, geisler_filename_distractor);

		    #define G4_LAYER
		    #ifdef G4_LAYER
			HyPerLayer * l4_geisler = new GeislerLayer("L4 Geisler", hc);
			LayerProbe * statsl4_geisler = new StatsProbe(BufActivity, "L4 Geisler :");
			l4_geisler->insertProbe(statsl4_geisler);

			HyPerConn * l3_geisler_l4_geisler =
				new CocircConn("L3 Geisler to L4 Geisler",  hc, l3_geisler, 	l4_geisler,
					CHANNEL_EXC);

			// const char * geisler4_filename_target = "./input/256/amoeba40K_G4/w13_last.pvp";
			HyPerConn * l3_geisler_l4_geisler_target =
				new KernelConn("L3 Geisler to L4 Geisler Target", hc, l3_geisler, l4_geisler,
					CHANNEL_INH, geisler_filename_target);
			// const char * geisler4_filename_distractor = "./input/256/distractor40K_G4/w13_last.pvp";
			HyPerConn * l3_geisler_l4_geisler_distractor =
				new KernelConn("L3 Geisler to L4 Geisler Distractor", hc, l3_geisler, l4_geisler,
					CHANNEL_INH, geisler_filename_distractor);

		    #endif G4_LAYER

        } // end else clause for if( training_flag )

        hc->run();

        /* clean up (HyPerCol owns layers and connections, don't delete them) */
        delete hc;


        return 0;
}
