%% Data collection and clean
clear all;
load('XXXXX');
RM = ismissing(REST); % list missing data (NaN) as 1, and others 0
REST_NEW = REST(~any(RM,2),:); % Clean the data so that a new dataset table by removing all rows containing missing data
% how to read the data only by calling the name of that colum: REST_NEW.W1
W1 = REST_NEW.W1;
W2 = REST_NEW.W2;
D = REST_NEW.W2 - REST_NEW.W1;
A = double(D > 0); 
%  A = 1 IF D > 0 , and = 0 otherwise
%  double function is to set logical array to numeric array
F1 = REST_NEW.EMPFT + 0.5*REST_NEW.EMPPT;
F2 = REST_NEW.EMPFT2 + 0.5*REST_NEW.EMPPT2;
Y = F2 - F1;
Z = REST_NEW.STATE; % Z = 1 if NJ, 0 if PA
X = [REST_NEW.CHAIN, REST_NEW.CO_OWNED];
% In CHAIN: 1 = bk; 2 = kfc; 3 = roys; 4 = wendys
% In CO_OWNED: 1 if company owned (co), and 0 if not owned (no)
X = X(:,1)*10 + X(:,2);
% X column as 8 distinct categories, and I defined them as
% 10(bk_no),11(bk_co), 20(kfc_no), 21(kfc_co), 30(roys_no), 31(roys_co),
% 40(wendys_no), and 41(wendys_co)

% Set a new table with 8 variables;
data = table(W1,W2,D,A,F1,F2,Y,Z,X);
save('restaurant','data');
%% 1) Potential Outcomes
% Y(0) and Y(1) are potential outcomes, indicating that change in wage will
% lead the change in employment

%% 2) Relevant Summary Statistics
%% 2.1 association between D and Y; between A and Y
clc;
clear all;
load('C:\Users\Ran\Desktop\hw2_2690\restaurant.mat');

% 1) Association between D and Y
DY = [data.D, data.Y];
myColors = zeros(size(data.D, 1), 3); % List of rgb colors for every data point.
% Look at matrix2 and determine what color each row should be.
rowsToSetBlue = data.Y > 0;
rowsToSetRed = data.Y <= 0;
% Set colormap to blue for the blue rows.
myColors(rowsToSetBlue, 1) = 0;
myColors(rowsToSetBlue, 2) = 0;
myColors(rowsToSetBlue, 3) = 1;
% Set colormap to red for the red rows.
myColors(rowsToSetRed, 1) = 1;
myColors(rowsToSetRed, 2) = 0;
myColors(rowsToSetRed, 3) = 0;

figure();
scatter(DY(:,2), DY(:,1),30, myColors)
title('Association between D and Y');
xlabel('Y: Change in Employment (F2 - F1)');
ylabel('D: Change in Wage (W2 - W1)');

% 2) Association between A and Y
figure();
boxplot(data.Y, data.A)
report = [mean(data.Y(data.A == 0)), std(data.Y(data.A == 0)),...
    mean(data.Y(data.A == 1)), std(data.Y(data.A == 1))];
report = array2table(report,'VariableNames',...
    {'MeanYA0','Std0','MeanYA1','Std1'})
report.Properties.RowNames = {'Report'}

title('Association between A and Y');
xlabel('A: 1 if D >0 (increase in wage)');
ylabel('Y: Change in Employment (F2 - F1)');
% a. Find the AY with A = 1, that W2 >W1
Y = double(data.Y > 0);

% Y > 0, increase in employment
% Y < 0, no change or decrease in employment
A1 = tabulate(Y(data.A == 1)); 
%Tabulate the number of employees from each category
A1(:,3) = A1(:,3)./100;
A1 = array2table(A1, 'VariableNames',{'Y','Count','Probability'})

% b. Find the AY with A = 0 meant that W2 <= W1
A0 = tabulate(Y(data.A == 0));
A0(:,3) = A0(:,3)./100;
A0 = array2table(A0, 'VariableNames',{'Y','Count','Probability'})
%% 2.2 association between D and X; between A and X
clc;
clear all;
load('C:\Users\Ran\Desktop\hw2_2690\restaurant.mat');

% 1) Association between D and X
% find the probability for each categorical for X
X = nominal(data.X);
getlevels(X);
figure;
scatter(data.X, data.D);
title('Association between D and X');
xlabel('X: Adjustment variables with 8 distinct categores');
ylabel('D: Change in Wage (W2 - W1)');
figure;
boxplot(data.D, X)
title('D, Grouped by X categories');
tabulate(X)

% 2) Association between A and X
AX = [data.A, data.X];
% a. Find the AX with A = 1, that W2 >W1 
% (To use function tabulate(X(data.A == 1)) can get the answer directly.
% And here, I wan to show how to get the answer step by step).
AX_1 = AX((AX(:,1) == 1),:);
AX1 = [unique(AX_1(:,2)) countcats(categorical(AX_1(:,2)))]; 
% AX1 is a matrix containning two columns: the first column is the 8 distinct
% categories of X, and the second column is the count occurences of each 
% category
pAX1 = []; % empty matrix for probabilities of distinct X under A = 1
for i = 1:8
    pAX1(i) = AX1(i,2)/sum(AX1(:,2));
end
pAX1 = [AX1 pAX1'];
pAX1 = array2table(pAX1, 'VariableNames',{'X','CountXA1','Probability'})
% array2table is to transfer array to table (table2array)

% b. Find the AX with A = 0, that W2 >W1
A0 = tabulate(X(data.A == 0));
A0 = cell2table(A0, 'VariableNames',{'X','CountXA0','Probability'});
A0.X = char(A0.X);
A0.Probability = A0.Probability/100;
pAX0 = A0

%% 2.3 conditional association between A and Y, given X
clc;
clear all;
load('C:\Users\Ran\Desktop\hw2_2690\restaurant.mat');

gX = [data.X, data.A, data.Y];
% Generate 8 matrices (g10~g41) with A and Y under given 8 distinct X
for i = [10 11 20 21 30 31 40 41];
    current = sortrows(gX((gX(:,1) == i),:),2);
    % sort each matric corresponding to A = 0 & 1 (2nd column) 
    % and other columns change based on the 2nd column 
    current = [current (current(:,3) > 0)]; 
    % the 4th column is to help us to determin whether Y 
    % (3rd column is increase or non-change/decrese
    eval(['g' num2str(i) '=current']);
end


% Generate 8 matrices (p10~p41) to give the probability of Y given A and X
for i = [10 11 20 21 30 31 40 41]
    eval(['test = g', num2str(i),';']); % test will be g10...g41 in the loop
    t1 =  test(test(:,2) == 1,:); % Under specific X and A = 1
    t0 = test(test(:,2) == 0,:); % under X= i and A = 0, 
    prob = [i,1,sum(test(:,2))/size(test,1),1,size(t1(t1(:,4) == 1,:),1)/size(t1,1); ...
        i,1,NaN,0, size(t1(t1(:,4) == 0,:),1)/size(t1,1);... 
        i,0,sum(test(:,2) == 0)/size(test,1),1,size(t0(t0(:,4) == 1,:),1)/size(t0,1); ...
        i,0, NaN,0, size(t0(t0(:,4) == 0,:),1)/size(t0,1)];
    % The 3rd column is to calculate the probablity of A given X
    % The 5th colum is to calculate the probablity of Y given X and A
    % t1/0(t1/0(:,4) == 1/0,:) is to find A = 1/0 under X = i
    % size function t to calculate the probability
    prob = array2table(prob, 'VariableNames',{'X','A','ProbA','Y','ProbY'});
    % covert array to table with titles
    eval(['p' num2str(i) '=prob']);
end

%% 3) Assume A is randomized. What is the estimate of tao under this assumption?
% tao = 0
% We want to estiimate tao = E(Y(1)-Y(0)), with E(Y(i)) = b0 + b1 * A(i).
% Next, E(Y(1)) - E(Y(0)) = (b0 + b1 * A(1)) - (b0 + b1 * A(0)) 
% = b1 * [A(1) - A(0)] = b1 * (1 - 0) = b1
% Thus, find the coefficient of A(i) in linear model or generalized linear
% model, and we can estimate of tao.

%% 4) rank-sum method and tao = 0
clc;
clear all;
load('C:\Users\Ran\Desktop\hw2_2690\restaurant.mat');
YA = [data.Y, data.A];
%YA = sortrows(YA,2);
x = YA(YA(:,2) == 1,1);
y = YA(YA(:,2) == 0,1);
[p,h,stats] = ranksum(x,y)

% p = 0.0032, and h = 1 indicate the regection of the null hypothesis of
% equal mean at the default 5% significant level.

%% 5) Now assume A is randomized within distinct levels of X
clc;
clear all;
load('C:\Users\Ran\Desktop\hw2_2690\restaurant.mat');

tao1 = 0; 
r1 = data.Y - tao1*data.A; % data conainted tao = 0
X = nominal(data.X); % set X to categorical data
table1 = table(r1,X);
fit1 = fitlm(table1,'r1~X');
resid1 = fit1.Residuals.Raw;
% fit1.Residuals is to give a table with 4 columns: Raw, Pearson,
% Studentized, and Standardiz. We need the Raw column for residuals, and
% here Raw: residual = y - yhat
fitted1 = r1 - resid1;
[p1,h1,stats1] = ranksum(resid1(data.A == 1), resid1(data.A == 0),'tail','both')
%p1 = 0.0069, h1 = 1; stats1 = zval: 2.7024 & ranksum: 53290
% The p1 = 0.0069 and h1 = 1 indicate that the ranksum rejects the null
% hyphothesis of tao = 0 (equal mean on treatment and control) at the
% default 5% significant level.

tao2 = -1; 
r2 = data.Y - tao2*data.A; % data conainted tao = 0
X = nominal(data.X); % set X to categorical data
table2 = table(r2,X);
fit2 = fitlm(table2,'r2~X');
resid2 = fit2.Residuals.Raw;
% fit1.Residuals is to give a table with 4 columns: Raw, Pearson,
% Studentized, and Standardiz. We need the Raw column for residuals, and
% here Raw: residual = y - yhat
fitted2 = r2 - resid2;
[p2,h2,stats2] = ranksum(resid2(data.A == 1), resid2(data.A == 0),'tail','both')
% p2 = 1.8361e-04 ; h2 = 1; stats2 = zval: 3.7406 & ranksum: 54109
% The p2 = 1.8361e-04 and h2 = 1 indicate that the ranksum rejects the null
% hyphothesis of tao = 0 (equal mean on treatment and control) at the
% default 5% significant level.

%% 6) Inverse probability weighting
%% 6.1 Distribution as Normal
clc;
clear all;
load('C:\Users\Ran\Desktop\hw2_2690\restaurant.mat');
X = categorical(data.X);
A = data.A;
Y = data.Y;

model1 = fitglm(X,A,'linear',...
    'distr','binomial');

res = model1.Residuals.Raw;
p_hat = A - res;
wt = A./p_hat + (1-A)./(1-p_hat);
ipw = fitglm(A,Y,'Weights',wt) %inverse probability weight
ci = coefCI(ipw)

%% 6.2 Distribution as Binomial 
clc;
clear all;
load('C:\Users\Ran\Desktop\hw2_2690\restaurant.mat');
X = categorical(data.X);
A = data.A;
Y = double(data.Y>0);
%Y = data.Y;
model1 = fitglm(X,A,'linear',...
    'distr','binomial');

res = model1.Residuals.Raw;
p_hat = A - res;
wt = A./p_hat + (1-A)./(1-p_hat);
ipw = fitglm(A,Y,'distr','binomial','Weights',wt) %inverse probability weight
ci = coefCI(ipw)

%% 7) Standardization method to generate the same
clc;
clear all;
load('C:\Users\Ran\Desktop\hw2_2690\restaurant.mat');

% A 'freq' matrix: 8 categories of X, and probability of occurrence
f = cell2table(tabulate(nominal(data.X)), ...
    'VariableNames',{'X','CountXA0','Probability'});
freq = f.Probability/100;
freq = [unique(data.X), freq];

Y0 = [];
j = freq(:,1)';
for i = freq(:,1)'
    Y0(find( j == i)) = ...
        mean(data.Y(data.A == 0 & data.X == i)) * freq(freq(:,1) == i,2);
end
Y0 = sum(Y0);

Y1 = [];
for i = freq(:,1)'
    Y1(find( j == i)) = ...
        mean(data.Y(data.A == 1 & data.X == i))* freq(freq(:,1) == i,2);
end
Y1 = sum(Y1);

tao = sum(Y1 - Y0)

Y0_var = 0;
for i = freq(:,1)'
    y0 = data.Y(data.A == 0 & data.X == i);
    Y0_var = Y0_var + var(y0)/length(y0)*freq(freq(:,1) == i,2)^2;
end

Y1_var = 0;
for i = freq(:,1)'
    y1 = data.Y(data.A == 1 & data.X == i);
    Y1_var = Y1_var + var(y1)/length(y1)*freq(freq(:,1) == i,2)^2;
end
ci = [-1,1];
ci = tao + 1.96*ci.*sqrt(Y1_var + Y0_var)
