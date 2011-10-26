function [connID, connIndex, num_arbors] = pvp_connectionID()

  global N_CONNECTIONS
  global SPIKING_FLAG
  global TRAINING_FLAG
  global NUM_ARBORS
  connIndex = struct;
  ij_conn = 0;

  if ( SPIKING_FLAG == 1 )
    
    N_CONNECTIONS = 19;
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
    connIndex.s1_c1 = ij_conn;
    connID{ 1, ij_conn } =  'S1toC1';

    ij_conn = ij_conn + 1;
    connIndex.s1_c1inh = ij_conn;
    connID{ 1, ij_conn } =  'S1toC1Inh';


    %% S1 Inh FF connections
    ij_conn = ij_conn + 1;
    connIndex.s1inh_s1 = ij_conn;
    connID{ 1, ij_conn } =  'S1InhtoS1';

    ij_conn = ij_conn + 1;
    connIndex.s1inh_s1inh_gap = ij_conn;
    connID{ 1, ij_conn } =  'S1InhtoS1InhGap';


    %% C1 connections
    ij_conn = ij_conn + 1;
    connIndex.c1_lgn = ij_conn;
    connID{ 1, ij_conn } =  'C1toLGN';

    ij_conn = ij_conn + 1;
    connIndex.c1_lgninh = ij_conn;
    connID{ 1, ij_conn } =  'C1toLGNInh';

    ij_conn = ij_conn + 1;
    connIndex.c1_c1 = ij_conn;
    connID{ 1, ij_conn } =  'C1toC1Lateral';

    ij_conn = ij_conn + 1;
    connIndex.c1_c1inh = ij_conn;
    connID{ 1, ij_conn } =  'C1toC1InhLateral';



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


    N_CONNECTIONS = ij_conn;
    num_arbors = repmat(1, [ 1, N_CONNECTIONS+1 ] );


  else % NON_SPIKING
    
    N_CONNECTIONS = 5;
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
    connIndex.l1_l1_lateral = ij_conn;
    connID{ 1, ij_conn } =  'L1ToL1Lateral';
    
    
    N_CONNECTIONS = N_CONNECTIONS + 2;
    connID = [connID, cell(1,2)];
    
    ij_conn = ij_conn + 1;
    connIndex.l2_l2_vertical = ij_conn;
    connID{ 1, ij_conn } =  'L2ToL2Vertical';
    
    ij_conn = ij_conn + 1;
    connIndex.l2_l2_lateral = ij_conn;
    connID{ 1, ij_conn } =  'L2ToL2Lateral';
    
    
    N_CONNECTIONS = N_CONNECTIONS + 2;
    connID = [connID, cell(1,2)];
    
    ij_conn = ij_conn + 1;
    connIndex.l3_l3_vertical = ij_conn;
    connID{ 1, ij_conn } =  'L3ToL3Vertical';
    
    ij_conn = ij_conn + 1;
    connIndex.l3_l3_lateral = ij_conn;
    connID{ 1, ij_conn } =  'L3ToL3Lateral';
    
    
    N_CONNECTIONS = N_CONNECTIONS + 2;
    connID = [connID, cell(1,2)];
    
    ij_conn = ij_conn + 1;
    connIndex.l4_l4_vertical = ij_conn;
    connID{ 1, ij_conn } =  'L4ToL4Vertical';
    
    ij_conn = ij_conn + 1;
    connIndex.l4_l4_lateral = ij_conn;
    connID{ 1, ij_conn } =  'L4ToL4Lateral';
    
    
    connID = [connID, cell(1,1)];
    
    num_arbors = repmat(1, [ 1, N_CONNECTIONS+1 ] );
    num_arbors(connIndex.l1_l1_lateral) = 2;
    num_arbors(connIndex.l2_l2_lateral) = 2;
    num_arbors(connIndex.l3_l3_lateral) = 2;
    num_arbors(connIndex.l4_l4_lateral) = 2;
    
    if TRAINING_FLAG == -1
      
      N_CONNECTIONS = 5;
      connIndex.l1_l1_ODD = 6;
      connID{ 1, 6 } =  'L1ToL1ODD';
      
    elseif TRAINING_FLAG == -2

      N_CONNECTIONS = 7;
      connIndex.l2_l2_ODD = 8;
      connID{ 1, 8 } =  'L12ToL12ODD';
      

    elseif TRAINING_FLAG == -3

      N_CONNECTIONS = 9;
      connIndex.l3_l3_ODD = 10;
      connID{ 1, 10 } =  'L13ToL13ODD';
      

    end%%if
    
  end%%if % spiking_flag