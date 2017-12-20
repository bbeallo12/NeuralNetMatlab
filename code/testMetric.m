format
clc
pop = 11;
a=rand(1,pop)
b=rand(1,pop)
buckets = 3;
buckSize = ceil(pop/buckets);
[buckets, buckSize]
[~,I1] = sort(a)
for i = 1:buckets
    if(i == buckets)
        It = I1((i-1)*buckSize+1:end);
        [~,I2] = sort(b(It));
        I2 = It(I2);
        I1((i-1)*buckSize+1:end) = I2
    else
        It = I1((i-1)*buckSize+1:i*buckSize);
        [~,I2] = sort(b(It));
        I2 = It(I2);
        I1((i-1)*buckSize+1:i*buckSize) = I2
    end
end