function savefile2(filename,image)

    image = uint8(image);
    imwrite(image, [filename '.png']);
    
end