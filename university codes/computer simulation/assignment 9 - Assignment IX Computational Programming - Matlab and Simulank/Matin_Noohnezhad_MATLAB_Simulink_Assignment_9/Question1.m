p = [5 0 3 -5 7];
root = roots(p);
deriv_3 = polyval(polyder(p),3);
integ = polyint(p);
Integral_1_to_3 = diff(polyval(integ,[1 3]));
counter = 1;
for i=-10:0.1:10
    x(counter) = i;
    y(counter) = polyval(p,i);
    counter = counter +1;
end
root
deriv_3
Integral_1_to_3
plot(x,y); grid on