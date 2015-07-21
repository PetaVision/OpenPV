function wordStruct = wnidToDefinition(structureXmlPath, wnid)
   wnidWords = wnidToWords(structureXmlPath, wnid);
   wnidGloss = wnidToGloss(structureXmlPath, wnid);
   wordStruct = struct('words', wnidWords , 'gloss', wnidGloss);
end
