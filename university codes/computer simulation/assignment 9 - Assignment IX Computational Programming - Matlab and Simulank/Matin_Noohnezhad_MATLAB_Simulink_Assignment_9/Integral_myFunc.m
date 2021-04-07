function Integral_myFunc(a,b,d)
% a = 1;
% b = 5;
sum = 0;
for i = a:d:b-d
    sum = sum + ((myFunc(i+d)+myFunc(i))*d/2);
end
sum