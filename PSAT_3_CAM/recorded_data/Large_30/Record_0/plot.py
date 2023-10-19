import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


fig=plt.figure()
df = pd.read_csv('3d_filtered.csv')
ax = plt.axes(projection="3d")
ax.plot3D(df['m0_x'], df['m0_y'], df['m0_z'])
ax.plot3D(df['m1_x'], df['m1_y'], df['m1_z'])
ax.plot3D(df['m2_x'], df['m2_y'], df['m2_z'])
ax.set_title('3D Position of Markers')
ax.set_xlabel('X - Axes')
ax.set_ylabel('Y - Axes')
ax.set_zlabel('Z - Axes')
ax.legend(('Marker 1', 'Marker 2', 'Marker 3'))

fig=plt.figure()
df = pd.read_csv('mid3D_filtered.csv')
ax = plt.axes(projection="3d")
ax.plot3D(df['x'], df['y'], df['z'])
ax.set_title('3D Mid Positions of Markers')
ax.set_xlabel('X - Axes')
ax.set_ylabel('Y - Axes')
ax.set_zlabel('Z - Axes')
plt.show()