function [Xinit] = Regression(F,opts)
    Omega  = opts.Omega;
    Out.Res=[]; Out.PSNRX1 = [];  Out.PSNRX2=[];  Out.PSNRX3=[]; Out.PSNRX4=[];
    
    %% Using Regression data as Initiation
    [~,~,b,~] = size(F);
    w = cell(b,3);
    X_m =mean(F(:,:,:,[4 5 6]),4);%三个无云时间的平均
    
    X_C = X_m.*Omega;        %无云时间无云部分
    F_C =F.*Omega;           %有云时间无云部分
    F_M = F.*(1-Omega);      %有云时间有云部分
    X_M = X_m.*(1-Omega);    %无云时间平均的有云部分
    
    for j = 1:b
        for k = 1:3
            w{j,k}= polyfit(X_C(:,:,j,k),F_C(:,:,j,k),1);
            F_M(:,:,j,k) = w{j,k}(1).*(X_M(:,:,j,k))+w{j,k}(2);
        end
    end
    Xinit = F_C+F_M;
    
    if isfield(opts, 'Xtrue')
        XT=opts.Xtrue;
        psnrx1 =  PSNR3D(255*Xinit(:,:,:,1),255*XT(:,:,:,1));
        psnrx2 =  PSNR3D(255*Xinit(:,:,:,2),255*XT(:,:,:,2));
        psnrx3 =  PSNR3D(255*Xinit(:,:,:,3),255*XT(:,:,:,3));
        psnrx4 =  PSNR3D(255*Xinit(:,:,:,1:3),255*XT(:,:,:,1:3));
        Out.PSNRX1 = [Out.PSNRX1, psnrx1]; Out.PSNRX2 = [Out.PSNRX2, psnrx2];
        Out.PSNRX3 = [Out.PSNRX3, psnrx3]; Out.PSNRX4 = [Out.PSNRX4, psnrx4];
    end
    fprintf('Regression: PSNR = %f,%f,%f,%f  \n',psnrx1,psnrx2,psnrx3,psnrx4);
end