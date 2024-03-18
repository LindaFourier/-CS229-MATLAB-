function g = sigmoid(z)
g = zeros(size(z));
g = (1 + exp(-1 *z)).^(-1);
end
