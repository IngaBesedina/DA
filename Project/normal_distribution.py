import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Создание объекта нормального распределения с заданными параметрами (среднее и стандартное отклонение)
mean = 150
std_dev = 8
normal_dist = norm(loc=mean, scale=std_dev)

# Генерация 1000 случайных величин, соответствующих нормальному распределению
random_values = normal_dist.rvs(size=1000)

# Рассчитываем плотность вероятности для значения x=154
value = 154
pdf_value = normal_dist.pdf(value)
print(f'Значение плотности вероятности для x={value}: {pdf_value}')

# Рассчитываем значение функции распределения для значения x=154
cdf_value = normal_dist.cdf(value)
print(f'Значение функции распределения для x={value}: {cdf_value}')

# Рассчитываем значение обратной функции распределения для вероятности p=0.95
p = 0.95
quantile_value = normal_dist.ppf(p)
print(f'Значение обратной функции распределения для p={p}: {quantile_value}')

# Вычисление вероятности того, что величина превысит 154
probability = normal_dist.sf(value)
print('Вероятность того, что случайная величина превысит 154:', probability)

# Задаем уровень доверия (1 - alpha)
confidence = 0.95

# Вычисляем доверительный интервал
lower_bound, upper_bound = norm.interval(confidence, loc=mean, scale=std_dev/np.sqrt(len(random_values)))

print(f'Доверительный интервал с точностью {confidence*100}%: ({lower_bound}, {upper_bound})')