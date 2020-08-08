from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np
y_true = [0.2896536913140272, 0.4213558786617638, 0.1389628406502325, 0.3994494315855385, 0.26472642385364226, 0.17561510568728747, 0.280815115479097, 0.3309763190160577, 0.3988931951233261, 0.4362394464952936]
y_pred = [0.26078856, 0.42918572, 0.25301203, 0.38770083, 0.28051493, 0.14381371, 0.28321767, 0.30873522, 0.28805557, 0.301353]
print('平均絶対誤差 (MAE)')
print(mean_absolute_error(y_true, y_pred))
print('平均二乗誤差 (MSE)')
print(mean_squared_error(y_true, y_pred))