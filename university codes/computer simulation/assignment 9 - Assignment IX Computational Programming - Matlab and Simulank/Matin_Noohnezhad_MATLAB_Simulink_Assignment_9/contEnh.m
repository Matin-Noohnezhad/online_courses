function contEnh
pic = imread('myPic.jpg');
pic = rgb2gray(pic);
width = 1000;
dim = size(pic);
pic = imresize(pic,[width*dim(1)/dim(2) width],'bicubic');
pic_imadjust = imadjust(pic);
pic_histeq = histeq(pic);
pic_adapthisteq = adapthisteq(pic);
subplot(2,2,1)
imshow(pic);
title('Original');
subplot(2,2,2)
imshow(pic_imadjust);
title('imadjust');
subplot(2,2,3)
imshow(pic_histeq);
title('histeq');
subplot(2,2,4)
imshow(pic_adapthisteq);
title('adapthisteq');
