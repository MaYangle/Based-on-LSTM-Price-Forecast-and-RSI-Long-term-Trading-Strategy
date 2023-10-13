clear;

data=csvread('data-rate.csv',1,0);
[m,n]=size(data);


gold = 0;
A=zeros(1,1000);
B=zeros(1,1000);
sunhao=0;
bitcoin = 0;
cash = 1000;

for j=1:1000
    for i =30:m
            zz=j*0.0001;
            bitupper=data(i,1)-zz;
            goldupper=data(i,2)-zz;
            
        if(bitupper>0&&goldupper>0)
            case1=(cash*0.98+bitcoin)*data(i,1)-cash*0.02;
            case2=(cash*0.99+gold)*data(i,1)-cash*0.1;
            
            if(case1-case2>0&&case1>0)
                bitcoin=bitcoin+cash*0.98;
                sunhao=sunhao+cash*0.02;
                
                if(cash~=0)
                    A(1,j)=A(1,j)+1;
                end
                cash=0;
            elseif(case2-case1>0&&case2>0)
                    gold=gold+cash*0.99;
                    sunhao=sunhao+cash*0.01;
                    
                    if(cash~=0)
                        B(1,j)=B(1,j)+1;
                    end
                    cash=0;
            end
            
        elseif(bitupper>0&&goldupper<0)
            case1=(cash*0.98+bitcoin+gold*0.99*0.98)*data(i,1)-(cash+gold*0.99)*0.02;
            if(case1>0)
                bitcoin=bitcoin+cash*0.98+gold*0.99*0.98;
                sunhao=sunhao+cash*0.02+gold*0.01+gold*0.99*0.02;
                
                if(cash~=0&&gold==0)
                    A(1,j)=A(1,j)+1;
                elseif(gold~=0)
                        A(1,j)=A(1,j)+1;
                        B(1,j)=B(1,j)+1;
                end
                cash=0;
                gold=0;
            else
                cash=cash+gold*0.99;
                sunhao=sunhao+gold*0.01;
                
                if(gold~=0)
                        B(1,j)=B(1,j)+1;
                end
                    gold=0;
            end
            
        elseif(goldupper>0&&bitupper<0)
            case1=(cash*0.99+gold+bitcoin*0.99*0.98)*data(i,2)-(cash+bitcoin*0.98)*0.01;
            if(case1>0)
                gold=gold+cash*0.99+bitcoin*0.99*0.98;
                sunhao=sunhao+cash*0.01+bitcoin*0.02+bitcoin*0.98*0.01;
                
                if(cash~=0&&bitcoin==0)
                    A(1,j)=A(1,j)+1;
                end
                if(bitcoin~=0)
                        B(1,j)=B(1,j)+1;
                end
                cash=0;
                bitcoin=0;
            else
                cash=cash+bitcoin*0.98;
                sunhao=sunhao+bitcoin*0.02;
                
                if(bitcoin~=0)
                    A(1,j)=A(1,j)+1;
                end
                bitcoin=0;
            end
        elseif(goldupper<0&&bitupper<0)
            cash=cash+bitcoin*0.98+gold*0.99;
            sunhao=sunhao+bitcoin*0.02+gold*0.01;
            
            if(bitcoin~=0)
                    A(1,j)=A(1,j)+1;
                end
            if(gold~=0)
                        B(1,j)=B(1,j)+1;
            end
            bitcoin=0;
            gold=0;
        end
        bitcoin=bitcoin*(1+data(i,1));
        if(data(i,2)==0)
            gold=gold;
        else
            gold=gold*(1+data(i,2));
        end
    end
    Cash(j)=cash+bitcoin*0.98+gold*0.99;
    Sunhao(j)=sunhao;
    bitcoin = 0;
    gold=0;
    cash = 1000;
end
figure
subplot(3,1,1)
plot(Cash)
subplot(3,1,2)
plot(Sunhao)
subplot(3,1,3)
plot(A)
hold on
plot(B)
hold off
