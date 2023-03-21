import datetime
import open3d as o3d
import pyransac3d as rnsc
import numpy as np
import matplotlib.cm as plt
import webcolors

class PointCloudProcessor:
    def __init__(self, path="/usr/local/zed/samples/mesh_gen.obj"):
        self.path = path
        self.mesh = o3d.io.read_triangle_mesh(path)
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = self.mesh.vertices
        self.segmented_pcd, self.plane_eq = self.segment_plane(self.pcd)
        self.clusters_arr = []
        self.labels = self.DBScan(self.segmented_pcd)
        self.max_label = self.labels.max()
        self.now = datetime.datetime.now()
        self.timestamp = self.now.strftime("%Y-%m-%d_%H-%M-%S")
    def remove_indexes(self, arr, indexes):
        mask = np.ones(arr.shape[0], dtype=bool)
        mask[indexes] = False
        new_arr = arr[mask]
        return new_arr

    def find_plane(self, np_points, th=0.02):
        Plane = rnsc.Plane()
        eq, inliers = Plane.fit(np_points, th)
        return inliers, eq

    def find_cuboid(self, np_points, th=0.02):
        Cuboid = rnsc.Cuboid()
        eq, inliers = Cuboid.fit(np_points, th)
        return inliers, eq

    def segment_plane(self, pcd):
        pcd_np = np.asarray(pcd.points)
        inliers, eq = self.find_plane(pcd_np)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        return outlier_cloud, eq

    def get_color_name(self, rgb):
        try:
            color_name = webcolors.rgb_to_name(rgb)
        except ValueError:
            color_name = self.get_closest_color(rgb)
        return color_name

    def get_closest_color(self, rgb):
        min_colors = {}
        for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - rgb[0]) ** 2
            gd = (g_c - rgb[1]) ** 2
            bd = (b_c - rgb[2]) ** 2
            min_colors[(rd + gd + bd)] = name
        return min_colors[min(min_colors.keys())]

    def Paint_PCD_as_labels(self,pcd2,labels):
        colors = plt.get_cmap("tab20")(labels / (self.max_label if self.max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd2.colors = o3d.utility.Vector3dVector(colors[:, :3])
        for i in range(self.max_label + 1):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) > 0:
                # Create point cloud object for current cluster label
                cluster_pcd = pcd2.select_by_index(cluster_indices)
                self.clusters_arr.append(cluster_pcd)
            color = colors[cluster_indices[0], :]
            color_name = plt.get_cmap("tab20")(i / (self.max_label if self.max_label > 0 else 1))
            color_name_new = (color_name[:3])
            color_tuple = tuple([x * 255 for x in color_name_new])
            print(f"Cluster {i}: Color index {i} - Color name: {self.get_color_name(color_tuple)}")
        return pcd2
    
    def DBScan(self,pcd2,eps = 0.08):
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(
                pcd2.cluster_dbscan(eps=eps , min_points=10, print_progress=True)) # eps=0.02
        return labels
    def Visualize(self,pcd):
        o3d.visualization.draw_geometries([pcd])
    def save_data(self):
        with open("data_of_cluster_%s.txt"%(self.timestamp), "w") as f:
            f.write("Volume of = "+ str(self.vol) +" m^3 \n")
            f.write("Surface Area = "+ str(self.surface_area) +" m^2 \n")
            f.write("Center Coordinates = "+ str(self.vol) +"\n")
    def main(self):
        self.painted_pcd = self.Paint_PCD_as_labels(self.segmented_pcd,self.labels)
        self.Visualize(self.painted_pcd)
        self.cluster_indx_inp = int(input("cluster?"))
        self.cluster_pcd = self.clusters_arr[self.cluster_indx_inp]
        self.hull, _ = self.cluster_pcd.compute_convex_hull()
        self.vol = self.hull.get_volume()
        self.cnt = self.hull.get_center()
        self.max_bound = self.hull.get_max_bound()
        self.min_bound = self.hull.get_min_bound()
        self.non_manifold_edges = self.hull.get_non_manifold_edges()
        self.non_manifold_vertices = self.hull.get_non_manifold_vertices()
        self.surface_area = self.hull.get_surface_area()
        self.hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(self.hull)
        self.hull_ls.paint_uniform_color((1, 0, 0))
        self.Visualize(self.cluster_pcd)

if __name__ == "__main__":
   feature_extractor = PointCloudProcessor()
   feature_extractor.main()
   feature_extractor.save_data()
