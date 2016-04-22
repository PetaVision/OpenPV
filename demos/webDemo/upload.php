<!doctype html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<title>PetaVision Image Classifier</title>
<link rel="icon" href="favicon.ico" type="image/x-icon" />
<!--<link rel="stylesheet" href="stylesheets/styles.css">-->
<script type="text/JavaScript">
var img;
function init(){
   var width = window.innerWidth * .98;
   var height = window.innerHeight * .98;
   var canvas = document.getElementById("canvas");
   var context = canvas.getContext("2d");
   img = new Image();
   img.onload = function() {

      if ((width/height) < (img.width/img.height)){
         height = (width/img.width) * img.height;
      } else if ((width/height) > (img.width/img.height)){
         width  = (height/img.height) * img.width;
      }
      
      canvas.setAttribute("width", width);
      canvas.setAttribute("height", height);
      context.drawImage(this, 0, 0, width, height);
      //TODO: Add resize listener to redraw canvas if browser is resized
   };
   drawMontage();
}
function drawMontage(){
   if(imageExists(url)){
      img.src = url + "?t=" + new Date().getTime(); // Arbitrary url added to avoid caching issues on some browsers.
      //img.src = url; // Some browsers (Chrome) seem to be smart enough to handle this. Others are not.
      setTimeout("drawMontage()", 500); //TODO: This is real dumb. Instead of reloading the image, check if modified/exists
   }
}
function imageExists(imgPath){
    var http = new XMLHttpRequest();
    //try{http.open('HEAD', imgPath, false); http.send();}catch{}
    http.open('HEAD', imgPath, false); 
    http.send();
    return http.status != 404;
}
</script>
</head>

<body onload="init();">
<?php
$target_dir = "uploads/";
$montagePath = "montage/";

if (!$_FILES["fileToUpload"]["error"] && isset($_POST["fileSubmit"])){
   $fileHash = md5_file($_FILES["fileToUpload"]["tmp_name"]) . time();
   $im = imagecreatefromstring(file_get_contents($_FILES["fileToUpload"]["tmp_name"])); 
}elseif (!empty($_POST["urlToUpload"] && isset($_POST["urlSubmit"]))){
   $fileHash = md5(file_get_contents($_POST["urlToUpload"])) . time(); 
   $im = imagecreatefromstring(file_get_contents($_POST["urlToUpload"]));
}else{exit("Error: No file selected.");}
$target_file = $target_dir . $fileHash . ".png";

if(!empty(glob($target_dir . "*.png"))){
   ob_implicit_flush(TRUE);
   echo "Please wait while a previous submission is being processed.";
   ob_flush();
}
while (!empty(glob($target_dir .  "*.png"))) {
   sleep(1);
}   
ob_flush();

if($im !== false) {
   imagepng($im, $target_file);
   imagedestroy($im);
   $montage = $montagePath . $fileHash . ".png";
   copy($montagePath . "montage.png", $montage); 
   /// Sets javascript varible url to the path to the montage image
   ?><script>var url = "<?php echo $montage ?>";</script><?php

   //$PV_CALL = "(cd PetaVision/demo/PASCAL_Classification && echo '" . getcwd() . "/" . $target_file . "' | mpiexec -np 6 Release/PASCAL_Classification -p paramsfiles/" . $_POST['model'] . ".params -t 5 -rows 3 -columns 2 2>&1 > webDemo.log)";
   //$PV_CALL = "(cd PetaVision/demo/PASCAL_Classification && echo '" . getcwd() . "/" . $target_file . "' | mpiexec -np 6 Release/PASCAL_Classification -p paramsfiles/" . $_POST['model'] . ".params -d 0,1,0,1,0,1 -t 5 -rows 3 -columns 2)";
   //$PV_CALL = "(cd PetaVision/projects/HeatMapLocalization && echo '" . getcwd() . "/" . $target_file . "' | mpirun -np 1 Release/HeatMapLocalization -p input/WEB_" . $_POST['model'] . ".params -t 2)";

   //$PV_CALL = "(cd PetaVision/projects/HeatMapLocalization && echo '" . getcwd() . "/" . $target_file . "' | mpirun -np 4 Release/HeatMapLocalization -p input/WEB_Crop_" . $_POST['model'] . ".params -d 0,1,0,1 -rows 1 -columns 4 -t 6 2>&1 > webDemo.log)";
   $PV_CALL = "(cd PetaVision/projects/HeatMapLocalization && echo '" . getcwd() . "/" . $target_file . "' | mpirun -np 4 Release/HeatMapLocalization -p input/WEB_Crop_" . $_POST['model'] . ".params -d 0,1,0,1 -rows 1 -columns 4 -t 6 2>&1 > webDemo.log)";
   

   exec($PV_CALL . " > /dev/null &"); 
   exec("/usr/bin/php monitor.php " . $target_file . " " . $fileHash . " " . $montagePath . " > /dev/null &");
}else{echo "<h3>This file is either corrupted or is not an image.</h3>";}
?>
<canvas id="canvas"/>
</body>
