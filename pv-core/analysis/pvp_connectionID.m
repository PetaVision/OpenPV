function [connID, connIndex, num_arbors] = pvp_connectionID()

  global N_CONNECTIONS
  global N_LAYERS
  global SPIKING_FLAG
  global TRAINING_FLAG
  global NUM_ARBORS
  connIndex = struct;
  ij_conn = 0;

  if ( SPIKING_FLAG == 1 )
    
    N_CONNECTIONS = 22;
    connID = cell(1,N_CONNECTIONS);


    %% retinal connections
    ij_conn = ij_conn + 1;
    connIndex.image_retina = ij_conn;
    connID{ 1, ij_conn } =  'imagetoRetina';

    ij_conn = ij_conn + 1;
    connIndex.r_lgn = ij_conn;
    connID{ 1, ij_conn } =  'RetinatoLGN';

    ij_conn = ij_conn + 1;
    connIndex.r_lgninhff = ij_conn;
    connID{ 1, ij_conn } =  'RetinatoLGNInhFF';


    %% LGN connections
    ij_conn = ij_conn + 1;
    connIndex.lgn_lgninh = ij_conn;
    connID{ 1, ij_conn } =  'LGNtoLGNInh';

    ij_conn = ij_conn + 1;
    connIndex.lgn_s1 = ij_conn;
    connID{ 1, ij_conn } =  'LGNtoS1';

    ij_conn = ij_conn + 1;
    connIndex.lgn_s1inhff = ij_conn;
    connID{ 1, ij_conn } =  'LGNtoS1InhFF';


    %% LGNInhFF connections
    ij_conn = ij_conn + 1;
    connIndex.lgninhff_lgn = ij_conn;
    connID{ 1, ij_conn } =  'LGNInhFFtoLGN';



    %% LGNInh connections
    ij_conn = ij_conn + 1;
    connIndex.lgninh_lgn = ij_conn;
    connID{ 1, ij_conn } =  'LGNInhtoLGN';


    %% S1 connections
    ij_conn = ij_conn + 1;
    connIndex.s1_s1 = ij_conn;
    connID{ 1, ij_conn } =  'S1toS1';

    ij_conn = ij_conn + 1;
    connIndex.s1_s1inh = ij_conn;
    connID{ 1, ij_conn } =  'S1toS1Inh';

    ij_conn = ij_conn + 1;
    connIndex.s1_c1 = ij_conn;
    connID{ 1, ij_conn } =  'S1toC1';

    ij_conn = ij_conn + 1;
    connIndex.s1_c1inh = ij_conn;
    connID{ 1, ij_conn } =  'S1toC1Inh';


    %% S1 Inh connections
    ij_conn = ij_conn + 1;
    connIndex.s1inh_s1 = ij_conn;
    connID{ 1, ij_conn } =  'S1InhtoS1';

    ij_conn = ij_conn + 1;
    connIndex.s1inh_s1inh = ij_conn;
    connID{ 1, ij_conn } =  'S1InhtoS1Inh';

    ij_conn = ij_conn + 1;
    connIndex.s1inh_s1inh_gap = ij_conn;
    connID{ 1, ij_conn } =  'S1InhtoS1InhGap';


    %% C1 connections
    ij_conn = ij_conn + 1;
    connIndex.c1_c1 = ij_conn;
    connID{ 1, ij_conn } =  'C1toC1Lateral';

    ij_conn = ij_conn + 1;
    connIndex.c1_c1inh = ij_conn;
    connID{ 1, ij_conn } =  'C1toC1InhLateral';

    ij_conn = ij_conn + 1;
    connIndex.c1_h1 = ij_conn;
    connID{ 1, ij_conn } =  'C1toH1';

    ij_conn = ij_conn + 1;
    connIndex.c1_h1inh = ij_conn;
    connID{ 1, ij_conn } =  'C1toH1Inh';



    %% C1 Inh connections
    ij_conn = ij_conn + 1;
    connIndex.c1inh_c1 = ij_conn;
    connID{ 1, ij_conn } =  'C1InhtoC1';

    ij_conn = ij_conn + 1;
    connIndex.c1inh_c1inh = ij_conn;
    connID{ 1, ij_conn } =  'C1InhtoC1Inh';

    ij_conn = ij_conn + 1;
    connIndex.c1inh_c1inh_gap = ij_conn;
    connID{ 1, ij_conn } =  'C1InhtoC1InhGap';


    %% H1 connections
    ij_conn = ij_conn + 1;
    connIndex.h1_lgn = ij_conn;
    connID{ 1, ij_conn } =  'H1toLGN';

    ij_conn = ij_conn + 1;
    connIndex.h1_lgninh = ij_conn;
    connID{ 1, ij_conn } =  'H1toLGNInh';

    ij_conn = ij_conn + 1;
    connIndex.h1_h1 = ij_conn;
    connID{ 1, ij_conn } =  'H1toH1Lateral';

    ij_conn = ij_conn + 1;
    connIndex.h1_h1inh = ij_conn;
    connID{ 1, ij_conn } =  'H1toH1InhLateral';

    %% H1 Inh connections
    ij_conn = ij_conn + 1;
    connIndex.h1inh_h1 = ij_conn;
    connID{ 1, ij_conn } =  'H1InhtoH1';

    ij_conn = ij_conn + 1;
    connIndex.h1inh_h1inh = ij_conn;
    connID{ 1, ij_conn } =  'H1InhtoH1Inh';

    ij_conn = ij_conn + 1;
    connIndex.h1inh_h1inh_gap = ij_conn;
    connID{ 1, ij_conn } =  'H1InhtoH1InhGap';



    N_CONNECTIONS = ij_conn;
    num_arbors = repmat(1, [ 1, N_CONNECTIONS+1 ] );


  else % NON_SPIKING
    
    N_CONNECTIONS = 6;
    connID = cell(1,N_CONNECTIONS);

    ij_conn = ij_conn + 1;
    connIndex.r_l1 = ij_conn;
    connID{ 1, ij_conn } =  'Image2R';
    
    ij_conn = ij_conn + 1;
    connIndex.r_l1 = ij_conn;
    connID{ 1, ij_conn } =  'R2L1';
    
    ij_conn = ij_conn + 1;
    connIndex.r_l1inh = ij_conn;
    connID{ 1, ij_conn } =  'R2L1Inh';
    
    ij_conn = ij_conn + 1;
    connIndex.l1_l1_vertical = ij_conn;
    connID{ 1, ij_conn } =  'L1ToL1Vertical';
    
    ij_conn = ij_conn + 1;
    connIndex.l1_l1_target = ij_conn;
    connID{ 1, ij_conn } =  'L1ToL1Target';
    
    
    ij_conn = ij_conn + 1;
    connIndex.l1_l1_distractor = ij_conn;
    connID{ 1, ij_conn } =  'L1ToL1Distractor';
    
    if N_LAYERS > 4
    N_CONNECTIONS = N_CONNECTIONS + 2;
    connID = [connID, cell(1,2)];
    
    ij_conn = ij_conn + 1;
    connIndex.l2_l2_vertical = ij_conn;
    connID{ 1, ij_conn } =  'L2ToL2Vertical';
    
    ij_conn = ij_conn + 1;
    connIndex.l2_l2_target = ij_conn;
    connID{ 1, ij_conn } =  'L2ToL2Target';
    
    ij_conn = ij_conn + 1;
    connIndex.l2_l2_distractor = ij_conn;
    connID{ 1, ij_conn } =  'L2ToL2Distractor';
    
    
    N_CONNECTIONS = N_CONNECTIONS + 2;
    connID = [connID, cell(1,2)];
    
    ij_conn = ij_conn + 1;
    connIndex.l3_l3_vertical = ij_conn;
    connID{ 1, ij_conn } =  'L3ToL3Vertical';
    
    ij_conn = ij_conn + 1;
    connIndex.l3_l3_target = ij_conn;
    connID{ 1, ij_conn } =  'L3ToL3Target';
    
    ij_conn = ij_conn + 1;
    connIndex.l3_l3_distractor = ij_conn;
    connID{ 1, ij_conn } =  'L3ToL3Distractor';
    
    
    N_CONNECTIONS = N_CONNECTIONS + 2;
    connID = [connID, cell(1,2)];
    
    ij_conn = ij_conn + 1;
    connIndex.l4_l4_vertical = ij_conn;
    connID{ 1, ij_conn } =  'L4ToL4Vertical';
    
    ij_conn = ij_conn + 1;
    connIndex.l4_l4_target = ij_conn;
    connID{ 1, ij_conn } =  'L4ToL4Target';
    
    ij_conn = ij_conn + 1;
    connIndex.l4_l4_distractor = ij_conn;
    connID{ 1, ij_conn } =  'L4ToL4Distractor';
    
    
    N_CONNECTIONS = N_CONNECTIONS + 2;
    connID = [connID, cell(1,2)];
    
    ij_conn = ij_conn + 1;
    connIndex.l5_l5_vertical = ij_conn;
    connID{ 1, ij_conn } =  'L5ToL5Vertical';
    
    ij_conn = ij_conn + 1;
    connIndex.l5_l5_target = ij_conn;
    connID{ 1, ij_conn } =  'L5ToL5Target';
    
    ij_conn = ij_conn + 1;
    connIndex.l5_l5_distractor = ij_conn;
    connID{ 1, ij_conn } =  'L5ToL5Distractor';
    
    
    connID = [connID, cell(1,1)];
    endif
    
    num_arbors = repmat(1, [ 1, N_CONNECTIONS+1 ] );
    num_arbors(connIndex.l1_l1_target) = 1;
    if N_LAYERS > 4
    num_arbors(connIndex.l2_l2_target) = 1;
    num_arbors(connIndex.l3_l3_target) = 1;
    num_arbors(connIndex.l4_l4_target) = 1;
    endif
    
    if TRAINING_FLAG == -1
      
      N_CONNECTIONS = 6;
      connIndex.l1_l1_ODD = 7;
      connID{ 1, 7 } =  'L1ToL1ODD';
      
    elseif TRAINING_FLAG == -2

      N_CONNECTIONS = 9;
      connIndex.l2_l2_ODD = 10;
      connID{ 1, 10 } =  'L12ToL12ODD';
      

    elseif TRAINING_FLAG == -3

      N_CONNECTIONS = 12;
      connIndex.l3_l3_ODD = 13;
      connID{ 1, 13 } =  'L13ToL13ODD';
      
    elseif TRAINING_FLAG == -4

      N_CONNECTIONS = 15;
      connIndex.l4_l4_ODD = 16;
      connID{ 1, 14 } =  'L14ToL14ODD';
      
    elseif TRAINING_FLAG == -5

      N_CONNECTIONS = 18;
      connIndex.l5_l5_ODD = 19;
      connID{ 1, 15 } =  'L15ToL15ODD';
      

    end%%if
    
  end%%if % spiking_flag