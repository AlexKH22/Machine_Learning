from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

xs = np.array([1, 2, 3, 4, 5], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6], dtype=np.float64)


def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
         ((mean(xs) * mean(xs)) - mean(xs * xs)))

    b = mean(ys) - m * mean(xs)

    return m, b


m, b = best_fit_slope_and_intercept(xs, ys)

print(m, b)

regression_line = [(m*x)+b for x in xs]

for predict_x in range(6, 8):
    predict_y = (m*predict_x)+b

    xs = np.append(xs, [predict_x])
    ys = np.append(ys, [predict_y])
    regression_line.append(predict_y)

plt.scatter(xs, ys, color='#003F72', label='data')
plt.plot(xs, regression_line, label='regression line')
plt.legend(loc=4)
plt.show()
