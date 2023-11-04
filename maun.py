import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Initialize the main window
root = tk.Tk()
root.title("K-means Clustering")

# Create data entry fields
data_frame = ttk.LabelFrame(root, text="Enter Data")
data_frame.grid(row=0, column=0, padx=10, pady=10, sticky="w")

data_label = ttk.Label(data_frame, text="Enter data points (comma separated):")
data_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
data_entry = ttk.Entry(data_frame, width=50)
data_entry.grid(row=0, column=1, padx=10, pady=5)

k_label = ttk.Label(data_frame, text="Enter the number of clusters (K):")
k_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
k_entry = ttk.Entry(data_frame, width=5)
k_entry.grid(row=1, column=1, padx=10, pady=5)

# Function to perform K-means clustering
def perform_clustering():
    data_str = data_entry.get()
    k_value = k_entry.get()

    try:
        data = [list(map(float, point.split(",")) ) for point in data_str.split(";")]
        k_value = int(k_value)

        if k_value <= 0:
            messagebox.showerror("Error", "Number of clusters (K) should be greater than 0.")
        else:
            kmeans = KMeans(n_clusters=k_value)
            kmeans.fit(data)
            labels = kmeans.labels_
            
            # Display the clustered data
            clustered_data = pd.DataFrame({'X': [point[0] for point in data], 'Y': [point[1] for point in data], 'Cluster': labels})
            clustered_data_frame = ttk.LabelFrame(root, text="Clustered Data")
            clustered_data_frame.grid(row=1, column=0, padx=10, pady=10, sticky="w")
            pd_table = ttk.Treeview(clustered_data_frame, columns=("X", "Y", "Cluster"))
            pd_table.heading("#1", text="X")
            pd_table.heading("#2", text="Y")
            pd_table.heading("#3", text="Cluster")
            pd_table.column("#1", width=100)
            pd_table.column("#2", width=100)
            pd_table.column("#3", width=100)

            for i in range(len(clustered_data)):
                pd_table.insert('', i, values=(clustered_data['X'][i], clustered_data['Y'][i], clustered_data['Cluster'][i]))
            
            pd_table.grid(row=0, column=0, padx=10, pady=10)

            # Visualize the clusters
            plt.figure(figsize=(6, 6))
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            for i in range(k_value):
                plt.scatter(data[np.where(labels == i)][:, 0], data[np.where(labels == i)][:, 1], s=50, c=colors[i], label=f'Cluster {i + 1}')
            
            plt.title("K-means Clustering")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend(loc='best')
            plt.show()

    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please enter valid data and K value.")

# Create the cluster button
cluster_button = ttk.Button(root, text="Cluster Data", command=perform_clustering)
cluster_button.grid(row=0, column=1, padx=10, pady=10)

# Start the GUI main loop
root.mainloop()
