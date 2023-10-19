import pickle, time
import matplotlib.pyplot as plt

with open('point.pkl', 'rb') as f:
    x=pickle.load(f)

fig=plt.figure()
ax=plt.axes()
fig.add_axes(ax)
ax.set_xlim(0, 1000)
ax.set_ylim(0, 700)


img=plt.imread('image/calib_img_client_1.tiff')
plt.imshow(img)

for i in range(48):
    plt.scatter(x[0,i,0],x[0,i,1])
    fig.show()


# for i in range(48):
#     plt.scatter(x[1,i,0],x[1,i,1])
#     fig.show()

# for i in range(48):
#     plt.scatter(x[2,i,0],x[2,i,1])
#     fig.show()
    
print('ok')