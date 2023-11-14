from sklearn.datasets import fetch_openml

print("start....")
mnist = fetch_openml('mnist_784', data_home='./mnist_data', as_frame=False)
print("end")