# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#data = pd.read_csv(r"E:\Hdrology\FINAL_DATA\Merged_CSV_data\Merged_CSV_data2-log2.csv")
data = pd.read_csv(r"E:\Hdrology\FINAL_DATA\AAAAAA\InputData-CSV.csv")
#x=data.loc[:,['Watershed Area (km2)','Slope','Mean Elevation of the Watershed (m)','Fraction of Forest Cover','Surface Water Storage Area (km2)','Urban Area (km2)','Rainfall-Runoff Correlation','Mean Daily Flow (m3/s)','Mean Annual Rainfall (mm)',
#               'Mean Maximum Temperature (K/day)','Mean Minimum Temperature (K/day)','Mean Solar Radiation (MJ/m2)','Mean Humidity','Mean Wind Speed (m/s)','Perimeter (km)','Gravelius_Index','Soil Depth (m)','Sand (%)','Silt (%)','Clay (%)']]
#x=data.loc[:,['Watershed Area (km2)','Fraction of Forest Cover','Urban Area (km2)','Rainfall-Runoff Correlation','Mean Minimum Temperature (K/day)','Perimeter (km)','Gravelius_Index','Soil Depth (m)','Sand (%)','Silt (%)']]
#x=data.loc[:,['Watershed Area (km2)','Slope','Mean Elevation of the Watershed (m)','Fraction of Forest Cover','Surface Water Storage Area (km2)', 'Urban Area (km2)','Perimeter (km)', 'Mean Annual Rainfall (mm)', 'Gravelius_Index','Soil Depth (m)']]
x=data.loc[:,['Watershed Area (km2)','Slope','Fraction of Forest Cover','Surface Water Storage Area (km2)','Urban Area (km2)', 'Mean Annual Rainfall (mm)']]
y=data.loc[:,'Q10']
############### Best Accuracy in RFR #################
x_train = x.iloc[[0,2,3,4,5,7,9,10,11,12,13,15,16,17,19,21,22,23,24,25,26,28,29,30,32,34,36,37,38]]
x_test = x.iloc[[1,6,8,14,18,20,27,31,33,35]]
y_train = y.iloc[[0,2,3,4,5,7,9,10,11,12,13,15,16,17,19,21,22,23,24,25,26,28,29,30,32,34,36,37,38]]
y_test = y.iloc[[1,6,8,14,18,20,27,31,33,35]]

'''x_train = x.iloc[:29, :]
x_test = x.iloc[29:, :]
y_train = y.iloc[:29]
y_test = y.iloc[29:]'''

'''############## better accuracy in case of RFR #########################################
x_train = x.iloc[[0,2,3,4,5,6,7,8,9,11,13,15,16,17,19,20,21,22,23,24,25,27,28,30,32,33,36,37,38]]
x_test = x.iloc[[1,10,12,14,18,26,29,31,34,35]]
y_train = y.iloc[[0,2,3,4,5,6,7,8,9,11,13,15,16,17,19,20,21,22,23,24,25,27,28,30,32,33,36,37,38]]
y_test = y.iloc[[1,10,12,14,18,26,29,31,34,35]]'''
'''
############## best accuracy in case of SVM #########################################
x_train = x.iloc[[0,1,2,3,4,5,7,8,9,10,1,13,14,15,16,20,22,23,24,25,26,27,28,29,30,31,32,33,34,36]]
x_test = x.iloc[[6,9,11,17,18,19,21,35,37,38]]
y_train = y.iloc[[0,1,2,3,4,5,7,8,9,10,1,13,14,15,16,20,22,23,24,25,26,27,28,29,30,31,32,33,34,36]]
y_test = y.iloc[[6,9,11,17,18,19,21,35,37,38]]'''


'''#########################  not cuurently use it ###########################################
x_train = x.iloc[[0,1,2,3,6,7,9,10,11,12,13,14,15,16,17,18,20,21,23,24,25,26,27,30,31,32,33,34,37]]
x_test = x.iloc[[4,5,8,19,22,28,29,35,36,38]]
y_train = y.iloc[[0,1,2,3,6,7,9,10,11,12,13,14,15,16,17,18,20,21,23,24,25,26,27,30,31,32,33,34,37]]
y_test = y.iloc[[4,5,8,19,22,28,29,35,36,38]]'''


from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor


####################   code for finding best parameters   #####################
rf=RandomForestRegressor()
# create regressor object
from sklearn.model_selection import RandomizedSearchCV

n_estimators= [300,400, 600, 800]
bootstrap= [True, False]
max_depth= [30,50, 70, 100,150]
max_features= ['auto', 'sqrt']
min_samples_leaf= [1, 2, 3]
min_samples_split= [2,3,4,5,6]

  # Create the random grid
random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
                                n_iter = 10, cv =3, verbose=2, 
                               random_state=42, n_jobs = -1)
# code for finding the best parameters
rf_random.fit(x_train.values, y_train.values.ravel())
print(rf_random.best_params_)
print("Best score is {}".format(rf_random.best_score_))

BestP = rf_random.best_params_


regressor = RandomForestRegressor(bootstrap=BestP['bootstrap'],
                                  max_features=BestP['max_features'],
                                  min_samples_leaf=BestP['min_samples_leaf'],
                                  min_samples_split=BestP['min_samples_split'],
                                  n_estimators=BestP['n_estimators'],                                 
                                  max_depth=BestP['max_depth'])



'''regressor = RandomForestRegressor(bootstrap=True,
                                  max_features='auto',
                                  min_samples_leaf=1,
                                  min_samples_split=2,
                                  n_estimators=400,
                                  random_state=20,                                  
                                  max_depth=50)'''


# fit the regressor with x and y data
regressor.fit(x_train.values, y_train.values.ravel())
y_pred = regressor.predict(x_test)

y_pred1 = regressor.predict(x_train)



#print(rf_random.feature_importances_)
Feature=pd.DataFrame({'Feature_names':x.columns,'Importances':regressor.feature_importances_})
print(Feature)
#df=y_test
#df.to_csv('y_test-q100',index=False)


'''# Calculate the point density
from scipy.stats import gaussian_kde
xy = np.vstack([y_test,y_pred])
z = gaussian_kde(xy)(xy)
fig, ax = plt.subplots()
sc=ax.scatter(y_test, y_pred, c=z, s=10)
ax.set_xlabel('y_test',fontsize=14)
ax.set_ylabel('y_pred',fontsize=14)
plt.colorbar(sc, label="Density_Estimation")
plt.show()'''



############################# calculate the scatter plot ############################
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



from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 Score:', metrics.r2_score(y_test, y_pred) )

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_pred1))
print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_pred1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred1)))
print('R2 Score:', metrics.r2_score(y_train, y_pred1) )



