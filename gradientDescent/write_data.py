import numpy as np

a = np.random.randint(100, size=(200, 2))
theta = [1, 2, 3, 4, 5]
offset = 4

for i in range(len(a)):
    noise = np.random.randint(10, size=1)[0]
    y = 0
    for j in range(len(a[0])):
        y += a[i][j] * theta[j]
    y += offset + noise
    with open('data.txt', 'a+') as f:
        for k in range(len(a[0])):
            f.write(str(a[i][k]) + ',')

        f.write(str(y)+'\n')
        f.close()

    print('第{}条数据:{},{},噪音:{}'.format(i+1, a[i, :], y, noise))
