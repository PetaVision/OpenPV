function [connID, connIndex] = pvp_connectionID()

  connIndex = struct;
  ij_conn = 0;
  global N_CONNECTIONS
  global SPIKING_FLAG
  global TRAINING_FLAG

  if ( SPIKING_FLAG == 1 )
    
  N_CONNECTIONS = 24;
  connID = cell(1,N_CONNECTIONS);


				% retinal connections
  ij_conn = ij_conn + 1;
  connIndex.r_lgn = ij_conn;
  connID{ 1, ij_conn } =  'Retina to LGN';

  ij_conn = ij_conn + 1;
  connIndex.r_lgninhff = ij_conn;
  connID{ 1, ij_conn } =  'Retina to LGNInhFF';

				% LGN connections
  ij_conn = ij_conn + 1;
  connIndex.lgn_lgninhff = ij_conn;
  connID{ 1, ij_conn } =  'LGN to LGNInhFF';

  ij_conn = ij_conn + 1;
  connIndex.lgn_lgninh = ij_conn;
  connID{ 1, ij_conn } =  'LGN to LGNInh';

  ij_conn = ij_conn + 1;
  connIndex.lgn_l1 = ij_conn;
  connID{ 1, ij_conn } =  'LGN to L1';

  ij_conn = ij_conn + 1;
  connIndex.lgn_l1inhff = ij_conn;
  connID{ 1, ij_conn } =  'LGN to L1InhFF';

				% LGNInhFF connections
  ij_conn = ij_conn + 1;
  connIndex.lgninhff_lgn = ij_conn;
  connID{ 1, ij_conn } =  'LGNInhFF to LGN';

  ij_conn = ij_conn + 1;
  connIndex.lgninhff_lgn_inhB = ij_conn;
  connID{ 1, ij_conn } =  'LGNInhFF to LGN InhB';

  ij_conn = ij_conn + 1;
  connIndex.lgninhff_lgninhff_exc = ij_conn;
  connID{ 1, ij_conn } =  'LGNInh to LGNInhFF Exc';

  ij_conn = ij_conn + 1;
  connIndex.lgninhff_lgninhff = ij_conn;
  connID{ 1, ij_conn } =  'LGNInhFF to LGNInhFF';

  ij_conn = ij_conn + 1;
  connIndex.lgninhff_lgninhff_inhB = ij_conn;
  connID{ 1, ij_conn } =  'LGNInhFF to LGNInhFF InhB';

  ij_conn = ij_conn + 1;
  connIndex.lgninhff_lgninh = ij_conn;
  connID{ 1, ij_conn } =  'LGNInhFF to LGNInh';

  ij_conn = ij_conn + 1;
  connIndex.lgninhff_lgninh_inhB = ij_conn;
  connID{ 1, ij_conn } =  'LGNInhFF to LGNInh InhB';

				% LGNInh connections
  ij_conn = ij_conn + 1;
  connIndex.lgninh_lgn = ij_conn;
  connID{ 1, ij_conn } =  'LGN Inh to LGN';

  ij_conn = ij_conn + 1;
  connIndex.lgninh_lgninh_exc = ij_conn;
  connID{ 1, ij_conn } =  'LGN Inh to LGN Inh Exc';

  ij_conn = ij_conn + 1;
  connIndex.lgninh_lgninh = ij_conn;
  connID{ 1, ij_conn } =  'LGN Inh to LGN Inh';


				% V1 connections
  ij_conn = ij_conn + 1;
  connIndex.l1_lgn = ij_conn;
  connID{ 1, ij_conn } =  'L1 to LGN';

  ij_conn = ij_conn + 1;
  connIndex.l1_lgninh = ij_conn;
  connID{ 1, ij_conn } =  'L1 to LGN FF';

  ij_conn = ij_conn + 1;
  connIndex.l1_l1 = ij_conn;
  connID{ 1, ij_conn } =  'L1 to L1';

  ij_conn = ij_conn + 1;
  connIndex.l1_l1inh = ij_conn;
  connID{ 1, ij_conn } =  'L1 to L1 Inh';


				% V1 Inh FF connections
  ij_conn = ij_conn + 1;
  connIndex.l1inhff_l1 = ij_conn;
  connID{ 1, ij_conn } =  'L1InhFF to L1';

  ij_conn = ij_conn + 1;
  connIndex.l1inhff_l1_inhB = ij_conn;
  connID{ 1, ij_conn } =  'L1InhFF to L1 InhB';

  ij_conn = ij_conn + 1;
  connIndex.l1inhff_l1inhff_exc = ij_conn;
  connID{ 1, ij_conn } =  'L1InhFF to L1InhFF Exc';

  ij_conn = ij_conn + 1;
  connIndex.l1inhff_l1inhff = ij_conn;
  connID{ 1, ij_conn } =  'L1InhFF to L1InhFF';

  ij_conn = ij_conn + 1;
  connIndex.l1inhff_l1ihff_inhB = ij_conn;
  connID{ 1, ij_conn } =  'L1InhFF to L1InhFF InhB';

  ij_conn = ij_conn + 1;
  connIndex.l1inhff_l1inh = ij_conn;
  connID{ 1, ij_conn } =  'L1InhFF to L1Inh';

  ij_conn = ij_conn + 1;
  connIndex.l1inhff_l1ih_inhB = ij_conn;
  connID{ 1, ij_conn } =  'L1InhFF to L1Inh InhB';


				% V1 Inh connections
  ij_conn = ij_conn + 1;
  connIndex.l1inh_l1 = ij_conn;
  connID{ 1, ij_conn } =  'L1Inh to L1';

  ij_conn = ij_conn + 1;
  connIndex.l1inh_l1inhff = ij_conn;
  connID{ 1, ij_conn } =  'L1Inh to L1Inh FF';

  ij_conn = ij_conn + 1;
  connIndex.l1inh_l1inh = ij_conn;
  connID{ 1, ij_conn } =  'L1Inh to L1Inh';

  ij_conn = ij_conn + 1;
  connIndex.l1inh_l1inh_exc = ij_conn;
  connID{ 1, ij_conn } =  'L1Inh to L1Inh Exc';

elseif TRAINING_FLAG

  N_CONNECTIONS = 3;
  connID = cell(1,N_CONNECTIONS);


  ij_conn = ij_conn + 1;
  connIndex.r_l1 = ij_conn;
  connID{ 1, ij_conn } =  'Retina to L1';

  ij_conn = ij_conn + 1;
  connIndex.r_l1inh = ij_conn;
  connID{ 1, ij_conn } =  'Retina to L1Inh';

  ij_conn = ij_conn + 1;
  connIndex.l1_l1 = ij_conn;
  connID{ 1, ij_conn } =  'L1 to L1';

else
  

  N_CONNECTIONS = 5;
  connID = cell(1,N_CONNECTIONS);


  ij_conn = ij_conn + 1;
  connIndex.r_l1 = ij_conn;
  connID{ 1, ij_conn } =  'Retina to L1';

  ij_conn = ij_conn + 1;
  connIndex.r_l1inh = ij_conn;
  connID{ 1, ij_conn } =  'Retina to L1Inh';

  ij_conn = ij_conn + 1;
  connIndex.l1_l1_geisler = ij_conn;
  connID{ 1, ij_conn } =  'L1 to L1 Geisler';

  ij_conn = ij_conn + 1;
  connIndex.l1_l1_geisler_target = ij_conn;
  connID{ 1, ij_conn } =  'L1 to L1 Geisler Target';

  ij_conn = ij_conn + 1;
  connIndex.l1_l1_geisler_distractor = ij_conn;
  connID{ 1, ij_conn } =  'L1 to L1 Geisler Distractor';


endif % spiking_flag