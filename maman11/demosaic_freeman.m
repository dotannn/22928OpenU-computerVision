function res = demosaic_freeman(mosaic, k)
    
    im=double(mosaic);
    
    M = size(im, 1);
    N = size(im, 2);

    red_mask = repmat([1 0; 0 0], M/2, N/2);
    green_mask = repmat([0 1; 1 0], M/2, N/2);
    blue_mask = repmat([0 0; 0 1], M/2, N/2);

    R=im.*red_mask;
    G=im.*green_mask;
    B=im.*blue_mask;

    % interpulation:
    GG = imfilter(G, [0 1 0; 1 4 1; 0 1 0]/4);
    BB = imfilter(B,[1,2,1; 2,4,2;1,2,1]/4);
    RR = imfilter(R,[1,2,1; 2,4,2;1,2,1]/4);
        
    %freeman improvement:
    RR2 = medfilt2(RR-GG,[k,k]) + GG;
    BB2 = medfilt2(BB-GG,[k,k]) + GG;
    
    res(:,:,1)=RR2; res(:,:,2)=GG; res(:,:,3)=BB2;
    
    res = uint8(res);
end