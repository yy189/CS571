scatter3(1, 1, 1, '+');
hold on;

x = [0, 0, 0, 0, 1, 1, 1];
y = [0, 0, 1, 1, 0, 0, 1];
z = [0, 1, 0, 1, 0, 1, 0];
scatter3(x, y, z);
hold on;

syms x y z;
A = [1, 1, 0.5];
B = [1, 0.5, 1];
C = [0.5, 1, 1];
D = [ones(4, 1), [[x, y, z]; A; B; C]];
detd = det(D);
disp(strcat('The plane equation is: ', char(detd), '= 0'))
z = solve(detd, z);
%plot3(1, 1, 0.5, '*', 1, 0.5, 1, '*', 0.5, 1, 1, '*');
hold on;
fmesh(z);


