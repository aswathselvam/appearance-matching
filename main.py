import matplotlib.pyplot as plt
import os
from rene.utils.loaders import ReneDataset
import cv2 
import numpy as np
import cupy as cp
from tqdm import tqdm
from datasets import load_dataset
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import BSpline
from scipy.interpolate import make_interp_spline
from pathlib import Path

'''
Implementation of the paper: Software Library for Appearance Matching(SLAM) SHREE K. NAYAR
Visual Learning and Recognition of 3-D Objects from Appearance SHREE K. NAYAR
'''

class Feature_extractor:
    ''' create N dimensional vector of pixel intensities.'''
    def __init__(self):
        pass
    def columns_stack(image):
        return image.T.reshape(-1)
    def rows_stack(image):
        return image.reshape(-1)

    def chess_board_sampler_stack(image):
        return image.T.reshape(-1)
    
    def salt_and_pepper_stack(image):
        return image.T.reshape(-1)

# Lazy load the dataset
os.system("rm -rf test/*.png")
eigenvectors_3d = []

dataset_name = "coil" # coil or rene

object_name = "cheetah"


images = []
features = []
image_size = 40


if dataset_name == "rene":
    rene = ReneDataset(input_folder="rene_dataset") # https://github.com/eyecan-ai/rene
    num_lights = len(rene[object_name])
    num_poses = len(rene[object_name][0])

    for l in tqdm(range(1)): #num_lights
        num_poses = len(rene[object_name][l])
    
        for i in range(30,49): #num_poses
            # To get a sample, you can do the following:
            #Rene Dataser loader:
            sample = rene[object_name][l][i]  # <- scene=cube, light_pose=18, camera_pose=36
            # Each sample contains [camera, image, pose, light] keys
            # To actually load an image you can do this:
            image = sample["image"]()  # <- Notice the `()` at the end!
            image = image[100:600, 400:900]
            images.append(image)

elif dataset_name == "coil":
        directory_path = Path("coil-100/coil-100") #https://www.kaggle.com/datasets/jessicali9530/coil100
        obj_id = 6
        for i in range(0,355,5):
            file = directory_path / f"obj{obj_id}__{i}.png"
            image = cv2.imread(file)
            images.append(image)

elif dataset_name == "coil-torch":
        directory_path = "tensorflow-examples/coil-20-cnn/images/training/pigbank/"
        files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        for i in range(len(files)):
            print(files[i])
            image = cv2.imread(files[i])
            images.append(image)



for i, image in enumerate(images):        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.blur(image, (3,3))
        image = cv2.resize(image, (image_size, image_size)) # Resize image PxQ=N

        # Normalize the image
        # l2_norm = np.linalg.norm(image)
        # print(l2_norm)
        # if l2_norm == 0:  # Avoid division by zero
        #     image = image
        # else:
        #     image = image / l2_norm
        #     # x=11

        # print(image.shape)

        '''Form  a feature vector:'''
        feature = image.T.reshape(-1) # Stack Columns vertically into a 1D vector,  \
        # N dimensional vector of pixel intensities. Treat each dimension together as a orthonormal basis.
        # This means, an image is a point in the N dimensional space.

        '''Interpretation of Template matching:'''
        # Correlation in Image space: It is the pixel wise sum of squared differences between the two images.
        # which is also equal to the L2 distance in the N Dimensional space.

        features.append(feature)

        # print(feature.shape)
        # And use the item as you wish
        # plt.imshow(image)   
        # plt.savefig(f"test/{i:02d}.png")
        cv2.imwrite(f"test/{i:02d}.png", image)

'''Dimensionality Reduction:'''
# If we ahve a set of points in 3D space, and they happen to form a plane,
# we can just define a plane with 2D vector to represent the points instead of using 3D vector.
# we define a orthonormal basis of dimension k ( k<N) to represent our image feature vector

'''Shift the data to the origin'''
# compute the mean of the features and subtract it from each feature.
features = np.array(features)
print("Features shape: ",features.shape)
mean = np.mean(features, axis=0)
features = features - mean

mean_image = mean.reshape(image_size, image_size).T
cv2.imwrite("mean.png", mean_image)


'''Find the principal compoents of the imges:'''
# 1st principal component(e1) shows maximum variance in the images.
# Aka the best fit line, formed with "Least squares fit", for the distribution of the images.
'''Project each image onto the principal components'''
#  p = e1.f
# Every image is represented with a single number p.
# covariance_matrix = np.cov(features, rowvar=False)

# # Step 3: Compute eigenvalues and eigenvectors
# eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# # Step 4: Sort eigenvectors by eigenvalues in descending order
# sorted_indices = np.argsort(eigenvalues)[::-1]
# principal_components = eigenvectors[:, sorted_indices]

# features_cp = cp.asarray(features)
# Step 2: Apply SVD
U, S, Vt = np.linalg.svd(features.T@features, full_matrices=False)

# Step 3: Extract the principal components (orthonormal basis)
principal_components = Vt

print("S", S.shape)
print("principal_components", principal_components.shape)
# print(S)
# np.set_printoptions(suppress=True, precision=2)
percentage_of_k_largest_eigenvalues = np.cumsum(S)/np.sum(S) * 100
index = np.searchsorted(percentage_of_k_largest_eigenvalues, 95, side='right')
print(np.round(percentage_of_k_largest_eigenvalues,2))
print(f"How many principal components are sufficient? {index}")

# print("Vt,", Vt)
plt.close("all")
plt.plot(S[:index])
plt.xlabel("Index") # Set your desired x-axis label here
plt.ylabel("Eigenvalue Magnitude") # Set your desired y-axis label here
plt.title("Magnitude of Eigenvalues that are needed to reconstruct 95% of the data") # You might also want a title
plt.savefig("eigenvalues.png")

for i in range(index):
    eigenvector_image = (principal_components[i]).reshape(image_size, image_size).T
    # print("np.min ",np.min(eigenvector_image))
    # print("np.max ",np.max(eigenvector_image))
    eigenvector_image = ( eigenvector_image - np.min(eigenvector_image) )/ np.max(eigenvector_image - np.min(eigenvector_image))
    eigenvector_image = eigenvector_image * 255
    # print(eigenvector_image)
    cv2.imwrite(f"eigen_image/eigenvector_{i}_image.png", eigenvector_image)




# p = features @ principal_components[:index]
print("principal_components.shape: ", principal_components.shape)
# k dimension projection of the features using k eigen vectors
k = 1000
projected_components = features @ principal_components[:,:k] # (M, N) * (N, k) = (M, k)
print("projected_components shape: ",projected_components.shape)
eigenvector_3d = principal_components[0,:3]
eigenvectors_3d.append(eigenvector_3d)




# Assuming `features` contains the feature vectors
# Reduce dimensionality to 3D using PCA (to align with eigenspace projection)
pca = PCA(n_components=3)
reduced_features = pca.fit_transform(projected_components)
print(reduced_features.shape)

# Sort points for interpolation
# sorted_indices = np.argsort(reduced_features[:, 0])
# sorted_points = reduced_features[sorted_indices]

# Extract coordinates
x = reduced_features[:, 0]
y = reduced_features[:, 1]
z = reduced_features[:, 2]

# Generate B-spline interpolation for manifold
t = np.linspace(0, 1, len(x))  # Parameterize by t
spl_x = make_interp_spline(t, x, k=3)
spl_y = make_interp_spline(t, y, k=3)
spl_z = make_interp_spline(t, z, k=3)

# Create fine grid for smooth manifold curve
t_fine = np.linspace(0, 1, 100)
manifold_x = spl_x(t_fine)
manifold_y = spl_y(t_fine)
manifold_z = spl_z(t_fine)

# Visualization of the manifold
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot original points
ax.scatter(x, y, z, color='blue', label='Projected Points')

# Plot manifold
ax.plot(manifold_x, manifold_y, manifold_z, color='red', linewidth=2, label='Manifold')

ax.set_title('3D Manifold Representation')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.legend()
plt.savefig("3d_manifold_representation.png") # You can change the filename and extension

plt.show()



'''Find the principal compoents E2 of the imges:'''
# 2nd principal component(e2) shows maximum variance in the images and perpendicular to e1.
# P = [p1 p2] = [e1 e2].f
# {e1, e2 e3 .. ek} is referred as Linear Subspace

'''Forward projection'''
# P = [p1 p2] = [e1 e2].f

'''Backward projection'''
# f = [e1 e2].P
