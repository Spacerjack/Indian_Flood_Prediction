# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv(r"E:\Hdrology\FINAL_DATA\Merged_CSV_data\Merged_CSV_data2-log2.csv")

x=data.loc[:,['Watershed Area (km2)','Slope','Mean Elevation of the Watershed (m)','Fraction of Forest Cover','Surface Water Storage Area (km2)', 'Urban Area (km2)','Perimeter (km)', 'Mean Annual Rainfall (mm)', 'Gravelius_Index','Soil Depth (m)']]
y=data.loc[:,'Q50']
x_train = x.iloc[[0,2,3,4,5,6,7,8,9,11,13,15,16,17,19,20,21,22,23,24,25,27,28,30,32,33,36,37,38]]
x_test = x.iloc[[1,10,12,14,18,26,29,31,34,35]]
y_train = y.iloc[[0,2,3,4,5,6,7,8,9,11,13,15,16,17,19,20,21,22,23,24,25,27,28,30,32,33,36,37,38]]
y_test = y.iloc[[1,10,12,14,18,26,29,31,34,35]]



from sklearn.ensemble import RandomForestRegressor


regressor = RandomForestRegressor(bootstrap=False,
                                  max_features='sqrt',
                                  min_samples_leaf=2,
                                  min_samples_split=3,
                                  n_estimators=300,
                                  random_state=20,                                  
                                  max_depth=100)


# fit the regressor with x and y data
regressor.fit(x_train.values, y_train.values.ravel())
y_pred = regressor.predict(x_test)




x_plot = y_test
y_plot = y_pred
fig, ax = plt.subplots()
_ = ax.scatter(x_plot, y_plot, c=x_plot, cmap='plasma')
z = np.polyfit(x_plot, y_plot, 1)
p = np.poly1d(z)
ax.set_xlabel('y_test',fontsize=14)
ax.set_ylabel('y_pred',fontsize=14)
plt.plot(x_plot, p(x_plot))
plt.show()


'''
# Calculate the point density
from scipy.stats import gaussian_kde
xy = np.vstack([y_test,y_pred])
z = gaussian_kde(xy)(xy)
fig, ax = plt.subplots()
sc=ax.scatter(y_test, y_pred, c=z, s=10)
ax.set_xlabel('y_test',fontsize=14)
ax.set_ylabel('y_pred',fontsize=14)
plt.colorbar(sc, label="Density_Estimation")
plt.show()'''
