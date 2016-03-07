<?php
$target_file = $argv[1];
$fileHash =    $argv[2];
$montagePath = $argv[3];
$montage = $montagePath . $fileHash . ".png";
/// While the final montage image has not been generated, look for intermediate ones and display them
$montageCount = 0;
while (!file_exists(glob($montagePath . $fileHash . "_*_final.tif")[0])) {
   $montageFiles = glob($montagePath . $fileHash . "_*[0-9].tif");
   if (count($montageFiles) > $montageCount) {
      sort($montageFiles, SORT_NATURAL);
      sleep(1);
      //exec("convert " . $montageFiles[$montageCount++] . "[0] " . $montage);
      exec("convert " . $montageFiles[$montageCount++] . "[0] " . $montage . "_tmp.png");
      //sleep(1);
      copy($montage."_tmp.png", $montage); // Stupid hack to deal with asynchrony 
   }
   sleep(1);
}
$montageFinal = glob($montagePath . $fileHash . "_*_final.tif")[0];
//copy($montageFinal, $montage); 
exec("convert " . $montageFinal . "[0] " . $montage);
//exec("convert " . $montageFinal . "[0] " . $montage . "_tmp.png");
//sleep(1);
//copy($montage."_tmp.png", $montage); // Stupid hack to deal with asynchrony 

/// Garbage Collection TODO: Potentially wait here to avoid race conditions
unlink($montageFinal);
$montageFiles = glob($montagePath . $fileHash . "_*[0-9].tif"); // Re-glob in case some were missed (Earlier race condition)
foreach ($montageFiles as $toClean) {
   unlink($toClean);
}
sleep(1); // To be sure js refreshed the last image
unlink($montage."_tmp.png");
rename($montage, "archive/" . $fileHash . ".png"); // For testing, mostly
unlink($target_file); // This should happen last because it prevents a duplicate upload collision
?>
