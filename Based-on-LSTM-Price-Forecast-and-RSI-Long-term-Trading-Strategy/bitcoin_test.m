clear;clc;

data=csvread('data-rate.csv',1,0);
[m,n]=size(data);


A=zeros(1,1000);

sunhao=0;
gold = 0;
cash = 1000;
for j=1:1000
    zengchang=j*0.0001;
    for i =30:m

            
        if(data(i,1)>zengchang)
            if(cash*0.02<((gold+cash*0.98)-gold))
                gold=gold+cash*0.98;
                sunhao=sunhao+cash*0.02;
                
                if(cash~=0)
                A(j)=A(j)+1;
                end
                cash=0;

            end
        end
        if(data(i,1)<0)
            cash=cash+gold*0.98;
            sunhao=sunhao+gold*0.02;
            
            if(gold~=0)
            A(j)=A(j)+1;
            end
            gold=0;
        end
        if(data(i,1)==0)
            gold=gold;
        else
            gold=gold*(1+data(i,1));
        end
        
    end
    Cash(j)=cash+gold*0.98;
    Sunhao(j)=sunhao;
    gold = 0;
    cash = 1000;
end
figure
subplot(3,1,1)
plot(Cash)
subplot(3,1,2)
plot(Sunhao)
subplot(3,1,3)
plot(A)