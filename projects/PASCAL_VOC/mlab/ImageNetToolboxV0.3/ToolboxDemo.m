% wnid you can use to test the function
% synset dagame, lemonwood tree  n12662074
% synset Bermuda onion           n07722390
% synset football game           n00468480
% synset contact sport           n00433458

homefolder = './';        %set homefolder to a path where you want to store the synset
% userName = 'username';   % a valid userName and accessKey
% accessKey = 'accesskey'; 
userName = 'username';
accessKey = 'accesskey';
wnid = 'n07722390'; 
isRecursive = 1;            

downloadImages(homefolder, userName, accessKey, wnid, isRecursive)

t=wnidToDefinition(fullfile(homefolder, 'structure_released.xml'), wnid)
