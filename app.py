from flask import Flask, render_template
from flask import request
app = Flask(__name__)

@app.route("/")
def main():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def my_form_get():
	import graphlab
	# In[23]:
	
	# In[36]:
	# In[37]:

	engsize = int(request.form.get("engine-size"))
	mile = int(request.form.get("mile"))
	width = int (request.form.get("width"))
	cname = str(request.form.get("cname"))
	hp = int(request.form.get("hp"))
	make =str(request.form.get("make"))
	
	

	

	html = "<html><body bgcolor='#00FF7F'><center><h1 >YOUR CAR EVALUATION RESULTS</h1></center>";
	html += "<p style='font-size:40px;'>Hi "+cname+"!<br>We congratulate you on your second hand car purchase.Your car's condition is reviewed and is provided in the table below."

	
	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	#get_ipython().run_line_magic('matplotlib', 'inline')

	col_names = ['symboling','normalized-losses','make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location','wheel-base','length','width','height','curb-weight','engine-type','num-of-cylinders','engine-size','fuel-system','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price']
	cars = pd.read_csv('imports-85.data', names=col_names)

	#cars.info()
	#cars.head()

	cars['normalized-losses'].value_counts()

	cars.drop('normalized-losses',axis = 1, inplace = True)

	cars.replace("?", np.nan, inplace=True)

	#cars.info()

	cars.dropna(axis = 0, how = 'any', inplace = True)

	cars[['price','peak-rpm','horsepower']] = cars[['price','peak-rpm','horsepower']].astype(int)
	cars[['stroke','bore',]] = cars[['stroke','bore',]].astype(float)

	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import StandardScaler
	from sklearn.neighbors import KNeighborsRegressor
	from sklearn.metrics import mean_squared_error

	def knn_train_test(df, col, kval):
		X = df[[col]]
		y = df['price']
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
		sc_X = StandardScaler()
		X_train = sc_X.fit_transform(X_train)
		X_test = sc_X.transform(X_test)
		for k in kval:
			knn = KNeighborsRegressor(k)
			knn.fit(X_train, y_train)
			y_pred = knn.predict(X_test)
			mse = mean_squared_error(y_test, y_pred)
			rmse = mse ** (1/2)
			#print("RMSE of {} for k = {} : {}".format(col,k, rmse))
			plt.xlim([0,10])
			plt.ylim([0,10000])
			plt.xticks([1,3,5,7,9])
			plt.bar(k,rmse)
							
		#plt.show()      

	features = ['highway-mpg', 'city-mpg', 'peak-rpm', 'horsepower', 'engine-size', 'curb-weight', 
				'width', 'length', 'height', 'wheel-base','bore', 'stroke']

	for i in features:
		knn_train_test(cars, i, [1, 3, 5, 7, 9])



	two_var = ['engine-size', 'horsepower']
	three_var = ['engine-size', 'horsepower','width' ] # For 3 variables
	four_var = ['engine-size', 'horsepower', 'width', 'highway-mpg'] # For 4 variables
	five_var = ['engine-size', 'horsepower', 'width', 'highway-mpg','city-mpg'] # for 5 variables

	def knn_train_test1(df, col, k):
		X = df[col]
		y = df['price']
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
		sc_X = StandardScaler()
		X_train = sc_X.fit_transform(X_train)
		X_test = sc_X.transform(X_test)
		knn = KNeighborsRegressor(k)
		knn.fit(X_train, y_train)
		y_pred = knn.predict(X_test)
		mse = mean_squared_error(y_test, y_pred)
		#print(col)
		#print(y_pred)
		rmse = mse ** (1/2)
		#print("RMSE of {} for k = {} : {}".format(col,k, rmse))

	knn_train_test1(cars,two_var, 3)
	knn_train_test1(cars, three_var, 3)
	knn_train_test1(cars, four_var, 3)
	knn_train_test1(cars, five_var, 3)

	kval = np.arange(1,26,1) 

	def knn_train_test2(df, col, kval):
		X = df[col]
		y = df['price']
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
		sc_X = StandardScaler()
		X_train = sc_X.fit_transform(X_train)
		X_test = sc_X.transform(X_test)
		for k in kval:
			knn = KNeighborsRegressor(k)
			knn.fit(X_train, y_train)
			y_pred = knn.predict(X_test)
			mse = mean_squared_error(y_test, y_pred)
			rmse = mse ** (1/2)
			#print("RMSE of {} for k = {} : {}".format(col,k, rmse))
			plt.xlim([0,25])
			plt.ylim([0,7000])
			plt.xticks(kval)
			plt.xlabel("K value")
			plt.ylabel("RMSE")
			plt.title("Error graph for each K value")
			plt.bar(k,rmse)
		#plt.show()

	knn_train_test2(cars, four_var, kval)

	def knn_train_testing(df, col, k,inputd):
		X = df[col]
		y = df['price']
		#print(X)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
		sc_X = StandardScaler()
		print(type(X_test))
	# print(Y_test)
		X = sc_X.fit_transform(X)
		inputd = sc_X.transform(inputd)
		
		knn = KNeighborsRegressor(k)
		knn.fit(X, y)
		y_pred = knn.predict(inputd)
		#mse = mean_squared_error(y_test, y_pred)
		#print(col,"\n")
		#print(y_pred,"\n")
		return y_pred
		#rmse = mse ** (1/2)
		#print("RMSE of {} for k = {} : {}".format(col,k, rmse))

	column=['engine-size','horsepower','width','highway-mpg']
	inputdata= [{'engine-size':engsize,'horsepower':hp,'width':width,'highway-mpg':mile}]
	inputd=pd.DataFrame(inputdata)
	#print(inputd)
	predictedvalue=knn_train_testing(cars, column, 4,inputd)
	predictedvalue = predictedvalue*(10);


	#car_model = graphlab.linear_regression.create(train_data,target = 'cost' , features = ['years' , 'mileage'])
	#val = car_model.predict(sales1)


	
	html +="<br><td><b>Make of the car:</b></td><br>"
	html+="<td>"+make+"</td></tr><br>"
	html +="<td><b>Name of the Customer:</b></td><br>"
	html+="<td>"+cname+"</td></tr><br>"
	html += "<td><b>Car Selling Price:</b></td><br>"
	html +="<td>Rs."+str(predictedvalue)+"/-</td></tr></table></center>"

	#print(cost)
	#print(val[0])
	print(html)
	return html

if __name__ == "__main__":
	app.debug=True
	app.run()