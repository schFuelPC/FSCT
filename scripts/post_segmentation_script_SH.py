import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.patches import Circle, PathPatch
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import LineModelND, CircleModel, ransac, EllipseModel
import mpl_toolkits.mplot3d.art3d as art3d
import math
import pandas as pd
from scipy import stats, spatial
import time
import warnings
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
from skimage.measure import LineModelND, CircleModel, ransac
import glob
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial.distance import euclidean
from math import sin, cos, pi
import random
import os
from sklearn.neighbors import NearestNeighbors
from tools import load_file, save_file, subsample_point_cloud, get_heights_above_DTM, cluster_dbscan
from scipy.interpolate import griddata
from fsct_exceptions import DataQualityError
import SFP
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import copy

warnings.filterwarnings("ignore", category=RuntimeWarning)


class PostProcessing:
    def __init__(self, parameters):
        self.post_processing_time_start = time.time()
        self.parameters = parameters
        self.filename = self.parameters["point_cloud_filename"].replace("\\", "/")
        self.output_dir = (
            os.path.dirname(os.path.realpath(self.filename)).replace("\\", "/")
            + "/"
            + self.filename.split("/")[-1][:-4]
            + "_FSCT_output/"
        )
        self.filename = self.filename.split("/")[-1]

        self.noise_class_label = parameters["noise_class"]
        self.terrain_class_label = parameters["terrain_class"]
        self.vegetation_class_label = parameters["vegetation_class"]
        self.cwd_class_label = parameters["cwd_class"]
        self.stem_class_label = parameters["stem_class"]
        print("Loading segmented point cloud...")
        self.point_cloud, self.headers_of_interest = load_file(
            self.output_dir + "segmented.las", headers_of_interest=["x", "y", "z", "red", "green", "blue", "label"]
        )
        self.point_cloud = np.hstack(
            (self.point_cloud, np.zeros((self.point_cloud.shape[0], 1)))
        )  # Add height above DTM column
        self.headers_of_interest.append("height_above_DTM")  # Add height_above_DTM to the headers.
        self.label_index = self.headers_of_interest.index("label")
        self.point_cloud[:, self.label_index] = (
            self.point_cloud[:, self.label_index] + 1
        )  # index offset since noise_class was removed from inference.
        self.plot_summary = pd.read_csv(self.output_dir + "plot_summary.csv", index_col=None)

    def make_DTM(self, crop_dtm=False):
        print("Making DTM...")
        # Run CSF filter

        # full_point_cloud_kdtree = spatial.cKDTree(self.point_cloud[:, :2])
        # terrain_kdtree = spatial.cKDTree(self.terrain_points[:, :2])
        # xmin = np.floor(np.min(self.terrain_points[:, 0])) - 3
        # ymin = np.floor(np.min(self.terrain_points[:, 1])) - 3
        # xmax = np.ceil(np.max(self.terrain_points[:, 0])) + 3
        # ymax = np.ceil(np.max(self.terrain_points[:, 1])) + 3
        # x_points = np.linspace(xmin, xmax, int(np.ceil((xmax - xmin) / self.parameters["grid_resolution"])) + 1)
        # y_points = np.linspace(ymin, ymax, int(np.ceil((ymax - ymin) / self.parameters["grid_resolution"])) + 1)
        # grid_points = np.zeros((0, 3))
        #
        # for x in x_points:
        #     for y in y_points:
        #         radius = self.parameters["grid_resolution"] * 3
        #         indices = terrain_kdtree.query_ball_point([x, y], r=radius)
        #         while len(indices) <= 100 and radius <= self.parameters["grid_resolution"] * 5:
        #             radius += self.parameters["grid_resolution"]
        #             indices = terrain_kdtree.query_ball_point([x, y], r=radius)
        #         if len(indices) >= 100:
        #             z = np.percentile(self.terrain_points[indices, 2], 20)
        #             grid_points = np.vstack((grid_points, np.array([[x, y, z]])))
        #
        #         else:
        #             indices = full_point_cloud_kdtree.query_ball_point([x, y], r=radius)
        #             while len(indices) <= 100:
        #                 radius += self.parameters["grid_resolution"]
        #                 indices = full_point_cloud_kdtree.query_ball_point([x, y], r=radius)
        #
        #             z = np.percentile(self.point_cloud[indices, 2], 2.5)
        #             grid_points = np.vstack((grid_points, np.array([[x, y, z]])))
        #
        # if self.parameters["plot_radius"] > 0:
        #     plot_centre = [[float(self.plot_summary["Plot Centre X"]), float(self.plot_summary["Plot Centre Y"])]]
        #     crop_radius = self.parameters["plot_radius"] + self.parameters["plot_radius_buffer"]
        #     grid_points = grid_points[np.linalg.norm(grid_points[:, :2] - plot_centre, axis=1) <= crop_radius]
        #
        # elif crop_dtm:
        #     distances, _ = full_point_cloud_kdtree.query(grid_points[:, :2], k=[1])
        #     distances = np.squeeze(distances)
        #     grid_points = grid_points[distances <= self.parameters["grid_resolution"]]
        # print("    DTM Done")
        return grid_points


    def process_point_cloud(self):
        #Lets try making an SFP object - then everything we do after this point can be in SFP language
        pc_file = r'E:\data\fsct_training\eric_carl\merged\clipped\tiled\760_transect\manual_noise_clean\tiles_cleaned_trimmed\full_transect_complied_CanVegCombined_SurfVegTerrainCombined\reference_classification_FSCT_output\segmented.las'
        pc = SFP.pointCloud(pc_file, adjust_ll=False)
        # pc = SFP.pointCloud((self.output_dir + "segmented.las"), adjust_ll = False)

        # Need to reclasify it, so we can work with the ground filter correctly
        pc = reclassify_FSCT_to_Hillman(pc)

        #Create a separate terrain object and separate out the ground
        terrain_obj = copy.deepcopy(pc)
        terrain_obj.points = terrain_obj.points.loc[terrain_obj.points.classification == 3,:]

        #Reduce the size of the pc object so that then you can merge it back in
        pc.points = pc.points.loc[pc.points.classification != 3, :]

        #Apply ground filter to terrain points
        grid_def = SFP.generate_grid_definition(0.04, pc.lowerleft - 1, grid_dims_m = pc.points[['x','y','z']].max() + 1 - pc.lowerleft)
        csf_arguments = dict(iterations=1000, cloth_resolution=0.18, rigidness=3, class_threshold=0.03, time_step=0.4)

        terrain_obj.apply_ground_filter(csf_arguments, type=3)

        #Need to bring the terrain back in then delete the terrain object from memory
        frames = [terrain_obj.points, pc.points]
        pc.points = pd.concat(frames)
        del terrain_obj

        # Reclassify the points that were rejected from the ground points and make them vegetation
        pc = reclassify_ground_to_Veg(pc)

        # Normalise the point cloud
        pc.generate_dtm_grid(grid_def)
        pc.normalise_points()

        # Lets deal with cwd because that's easy
        # Reclassify cwd points that are above 3m to vegetation classification (4)
        cwd_idx = np.logical_and(pc.points.classification == 6, pc.points.agh > 3)
        pc.points.loc[cwd_idx, 'classification'] = 4

        # Lets deal with vegetation and stems difficulty
        # Start to identify the stem clumps that are sitting in trees
        # Once you identify the clump - look at surrounding neighbours
        # If they are vegetation, then make that point vegetation

        #Create a copy first and just deal with veg and stems
        veg_stems_obj = copy.deepcopy(pc)
        # veg_stems_idx = np.logical_or(veg_stems_obj.points.classification == 4, veg_stems_obj.points.classification == 5)
        # veg_stems_obj.points = veg_stems_obj.points.loc[veg_stems_idx, :]
        #Try just the stems
        stems_idx = veg_stems_obj.points.classification == 5
        veg_stems_obj.points = veg_stems_obj.points.loc[stems_idx, :]

        # Look at the noise calculation and see if that might give some insight as to how to do this:
        number_neighbours = 20
        voxel_size = 0.025
        std_deviation = 0.25

        v_space_vec = veg_stems_obj.create_voxel_space(voxel_size, generate_count=True)
        veg_stems_obj.voxel_space['drop_me'] = False
        veg_stems_obj.voxel_space['mean_neighbour'] = 0

        tree = KDTree(veg_stems_obj.voxel_space[['x_vox_ind', 'y_vox_ind', 'z_vox_ind']])
        nearest_dist, nearest_ind = tree.query(veg_stems_obj.voxel_space[['x_vox_ind', 'y_vox_ind', 'z_vox_ind']],
                                               k=number_neighbours)
        veg_stems_obj.voxel_space.loc[:, 'mean_neighbour'] = nearest_dist.mean(axis=1)
        veg_stems_obj.voxel_space.drop_me = veg_stems_obj.voxel_space.mean_neighbour > (
                    veg_stems_obj.voxel_space.mean_neighbour.mean() + std_deviation * veg_stems_obj.voxel_space.mean_neighbour.std())

        v_space_vec[:] = 1
        v_space_vec[veg_stems_obj.voxel_space.x_vox_ind,
                    veg_stems_obj.voxel_space.y_vox_ind,
                    veg_stems_obj.voxel_space.z_vox_ind] = veg_stems_obj.voxel_space.drop_me

        noise_ind = v_space_vec[veg_stems_obj.points.x_vox_ind, veg_stems_obj.points.y_vox_ind, veg_stems_obj.points.z_vox_ind]
        veg_stems_obj.points.loc[noise_ind == 1, 'noise'] = True
        veg_stems_obj.points.loc[noise_ind == 0, 'noise'] = False

        print(np.unique(veg_stems_obj.points.noise, return_counts=True))

        #Write this out and see what it looks like
        veg_stems_obj.points.loc[veg_stems_obj.points.noise == True, 'classification'] = 1.0
        veg_stems_obj.points.loc[veg_stems_obj.points.noise == False, 'classification'] = 2.0


        veg_stems_obj.write_las_file('test.las', 'E:/data/fsct_training/eric_carl/merged/clipped/tiled/760_transect/manual_noise_clean/tiles_cleaned_trimmed/full_transect_complied_CanVegCombined_SurfVegTerrainCombined/reference_classification_FSCT_output/',ignore_noise=False)




        self.terrain_points = self.point_cloud[
            self.point_cloud[:, self.label_index] == self.terrain_class_label
        ]  # -2 is now the class label as we added the height above DTM column.

        try:
            self.DTM = self.make_DTM(crop_dtm=True)
        except ValueError:
            raise DataQualityError("Failed to make DTM. \nThere probably aren't any terrain_points.")

        save_file(self.output_dir + "DTM.las", self.DTM)

        self.convexhull = spatial.ConvexHull(self.DTM[:, :2])
        self.plot_area = self.convexhull.volume / 10000  # volume is area in 2d.
        print("Plot area is approximately", self.plot_area, "ha")

        above_and_below_DTM_trim_dist = 0.2

        self.point_cloud = get_heights_above_DTM(
            self.point_cloud, self.DTM
        )  # Add a height above DTM column to the point clouds.
        self.terrain_points = self.point_cloud[self.point_cloud[:, self.label_index] == self.terrain_class_label]
        self.terrain_points_rejected = np.vstack(
            (
                self.terrain_points[self.terrain_points[:, -1] <= -above_and_below_DTM_trim_dist],
                self.terrain_points[self.terrain_points[:, -1] > above_and_below_DTM_trim_dist],
            )
        )
        self.terrain_points = self.terrain_points[
            np.logical_and(
                self.terrain_points[:, -1] > -above_and_below_DTM_trim_dist,
                self.terrain_points[:, -1] < above_and_below_DTM_trim_dist,
            )
        ]

        save_file(
            self.output_dir + "terrain_points.las",
            self.terrain_points,
            headers_of_interest=self.headers_of_interest,
            silent=False,
        )
        self.stem_points = self.point_cloud[self.point_cloud[:, self.label_index] == self.stem_class_label]
        self.terrain_points = np.vstack(
            (
                self.terrain_points,
                self.stem_points[
                    np.logical_and(
                        self.stem_points[:, -1] >= -above_and_below_DTM_trim_dist,
                        self.stem_points[:, -1] <= above_and_below_DTM_trim_dist,
                    )
                ],
            )
        )
        self.stem_points_rejected = self.stem_points[self.stem_points[:, -1] <= above_and_below_DTM_trim_dist]
        self.stem_points = self.stem_points[self.stem_points[:, -1] > above_and_below_DTM_trim_dist]
        save_file(
            self.output_dir + "stem_points.las",
            self.stem_points,
            headers_of_interest=self.headers_of_interest,
            silent=False,
        )

        self.vegetation_points = self.point_cloud[self.point_cloud[:, self.label_index] == self.vegetation_class_label]
        self.terrain_points = np.vstack(
            (
                self.terrain_points,
                self.vegetation_points[
                    np.logical_and(
                        self.vegetation_points[:, -1] >= -above_and_below_DTM_trim_dist,
                        self.vegetation_points[:, -1] <= above_and_below_DTM_trim_dist,
                    )
                ],
            )
        )
        self.vegetation_points_rejected = self.vegetation_points[
            self.vegetation_points[:, -1] <= above_and_below_DTM_trim_dist
        ]
        self.vegetation_points = self.vegetation_points[self.vegetation_points[:, -1] > above_and_below_DTM_trim_dist]
        save_file(
            self.output_dir + "vegetation_points.las",
            self.vegetation_points,
            headers_of_interest=self.headers_of_interest,
            silent=False,
        )

        self.cwd_points = self.point_cloud[
            self.point_cloud[:, self.label_index] == self.cwd_class_label
        ]  # -2 is now the class label as we added the height above DTM column.
        self.terrain_points = np.vstack(
            (
                self.terrain_points,
                self.cwd_points[
                    np.logical_and(
                        self.cwd_points[:, -1] >= -above_and_below_DTM_trim_dist,
                        self.cwd_points[:, -1] <= above_and_below_DTM_trim_dist,
                    )
                ],
            )
        )

        self.cwd_points_rejected = np.vstack(
            (
                self.cwd_points[self.cwd_points[:, -1] <= above_and_below_DTM_trim_dist],
                self.cwd_points[self.cwd_points[:, -1] >= 10],
            )
        )
        self.cwd_points = self.cwd_points[
            np.logical_and(self.cwd_points[:, -1] > above_and_below_DTM_trim_dist, self.cwd_points[:, -1] < 3)
        ]
        save_file(
            self.output_dir + "cwd_points.las",
            self.cwd_points,
            headers_of_interest=self.headers_of_interest,
            silent=False,
        )

        self.terrain_points[:, self.label_index] = self.terrain_class_label
        self.cleaned_pc = np.vstack((self.terrain_points, self.vegetation_points, self.cwd_points, self.stem_points))
        save_file(
            self.output_dir + "segmented_cleaned.las", self.cleaned_pc, headers_of_interest=self.headers_of_interest
        )
        save_file(
            self.output_dir + "terrain_points_rejected.las", self.terrain_points_rejected, headers_of_interest=self.headers_of_interest
        )

        self.post_processing_time_end = time.time()
        self.post_processing_time = self.post_processing_time_end - self.post_processing_time_start
        print("Post-processing took", self.post_processing_time, "seconds")
        self.plot_summary["Post processing time (s)"] = self.post_processing_time
        self.plot_summary["Num Terrain Points"] = self.terrain_points.shape[0]
        self.plot_summary["Num Vegetation Points"] = self.vegetation_points.shape[0]
        self.plot_summary["Num CWD Points"] = self.cwd_points.shape[0]
        self.plot_summary["Num Stem Points"] = self.stem_points.shape[0]
        self.plot_summary["Plot Area"] = self.plot_area
        self.plot_summary["Post processing time (s)"] = self.post_processing_time
        self.plot_summary.to_csv(self.output_dir + "plot_summary.csv", index=False)
        print("Post processing done.")

def reclassify_FSCT_to_Hillman(pc_obj):
    pc_obj.points['classification'] = pc_obj.points['classification'].replace(3,5)  # Stems classification from 3 to 5
    pc_obj.points['classification'] = pc_obj.points['classification'].replace(1,4)  # Vegetation classificaiton from 1 to 4
    pc_obj.points['classification'] = pc_obj.points['classification'].replace(2,6)  # CWD classification from 2 to 6
    pc_obj.points['classification'] = pc_obj.points['classification'].replace(0,3)  # Terrain classification from 0 to 3

    return pc_obj

def reclassify_ground_to_Veg(pc_obj):
    pc_obj.points['classification'] = pc_obj.points['classification'].replace(0,4)  # Reclassify rejected terrain from 0 to 4

    return pc_obj
