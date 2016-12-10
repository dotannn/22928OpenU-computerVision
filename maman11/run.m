%% gaussian blur:
img = imread('coloredImage.JPG');

imgGray = rgb2gray(img);

imgGrayCrop = imgGray(100:500,150:600);

imgGrayCropNoise = imnoise(imgGrayCrop,'gaussian',0,3);
H = fspecial('gaussian', 3);
res = imfilter(imgGrayCropNoise,H,'replicate');

mse = 1-ssim(res,imgGrayCrop)
mse2 = 1-ssim(imgGrayCropNoise,imgGrayCrop)


%% freeman section
close all;

img = imread('thirdAlternative.JPG');

imgMosaic = zeros([size(img,1) size(img,2)]);

imgMosaic(1:2:end,1:2:end) =img(1:2:end,1:2:end,1) ;
imgMosaic(1:2:end,2:2:end)= img(1:2:end,2:2:end,2) ;
imgMosaic(2:2:end,1:2:end) =  img(2:2:end,1:2:end,2);
imgMosaic(2:2:end,2:2:end) = img(2:2:end,2:2:end,3) ;

imgMosaic = uint8(imgMosaic);

kernels = [1,3,5,7,9,11];
MSEs=[0,0,0,0,0,0];
VARs=[0,0,0,0,0,0];
MAXs=[0,0,0,0,0,0];
figure,
for i=1: length(kernels)
    res = demosaic_freeman(imgMosaic,kernels(i));
    seColor = (double(res) - double(img)) .^2;
    subplot(2,6,i), imshow(res)
    MSEs(i) = mean(seColor(:));
    MAXs(i) = max(sqrt(seColor(:)));
    VARs(i) = var(sqrt(seColor(:)));
end

subplot(2,6,7:8), plot(kernels,MSEs), title('mse vs kernel size');
subplot(2,6,9:10), plot(kernels,VARs), title('var vs kernel size');
subplot(2,6,11:12), plot(kernels,MAXs), title('max vs kernel size');