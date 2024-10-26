import random
import matplotlib.pyplot as plt
import numpy as np

# Кількість значень, які будемо генерувати
num_samples = 10000
x_values = []

for _ in range(num_samples):
    # Генеруємо випадкове число u від 0 до 1
    u = random.uniform(0, 1)

    # Визначаємо інтервал і обчислюємо x
    if u < 0.3:
        # Інтервал від 0 до 2
        x = u / 0.15
    elif u < 0.7:
        # Інтервал від 2 до 4
        x = 2 + (u - 0.3) / 0.2
    elif u < 0.9:
        # Інтервал від 4 до 6
        x = 4 + (u - 0.7) / 0.1
    else:
        # Інтервал від 6 до 10
        x = 6 + (u - 0.9) / 0.025
    
    x_values.append(x)

# Побудова гістограми
plt.hist(x_values, bins=50, edgecolor='black', density=True)

# Налаштування графіка
plt.title('Гістограма згенерованих значень x')
plt.xlabel('x')
plt.ylabel('Щільність ймовірності')
plt.grid(True)

# Відображення графіка
plt.show()
