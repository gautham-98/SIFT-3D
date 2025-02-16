from typing import Tuple
import numpy as np
from PIL import Image
from numba import jit, prange
import io
import trimesh

from profiler import Profiler

profiler = Profiler()
class IcosphereSceneGenerator:

    class _CameraManager:
        """
        A class to hold the camera viewpoints, and a function to return the transformation from that viewpoint
        """
        @profiler.time_it
        def __init__(self, icosphere_radius:float, icosphere_subdivisions:float, resolution:Tuple[int,int], fov:Tuple[float, float]):
            self.camera = trimesh.scene.Camera(name="ico_cam", 
                                           resolution=resolution, 
                                           fov=fov)
            self.view_points = trimesh.creation.icosphere(radius=icosphere_radius, subdivisions=icosphere_subdivisions).vertices 

        @staticmethod
        @profiler.time_it 
        def get_camera_transformation(position:np.ndarray, target=np.array([0,0,0])):
            direction_vec = target - position
            z = direction_vec/np.linalg.norm(direction_vec)

            up = np.array([0,1,0]) if not np.all(np.absolute(z)==np.array([0,1,0])) else np.array([0,0,1])
            
            x = np.cross(up, z) 
            x = x/np.linalg.norm(x) 
            y = np.cross(z,x)

            # camera pointed along -z, transformation matrix
            transformation = np.eye(4)
            transformation[:3,:3] = np.stack((x,y,-z), axis=1)
            transformation[:3,3] = position
            return transformation
            

    class _SceneManager:
        """
        create a simple trimesh scene with the obj from filepath in the origin
        """
        @profiler.time_it
        def __init__(self, filepath:str):
            self.mesh = self._centre_mesh(trimesh.load(filepath))
            self.mesh = self.mesh.simplify_quadric_decimation(face_count=100)
            self.scene = trimesh.Scene(geometry={"main_object": self.mesh})

        @staticmethod
        def _centre_mesh(mesh): return mesh.apply_translation(-mesh.centroid)

    
    class _PixelTo3DMapper:
        """
        maps a pixel in image plane to world coordinate of the captured object
        """
        @profiler.time_it
        def __init__(self, scene:trimesh.Scene):
            self.scene = scene
            self.mesh = scene.geometry.get("main_object")
            self.mesh_triangles = self.mesh.vertices[self.mesh.faces]
        
        def get_map(self):
            pixels, origins, directions, mesh_triangles = self._get_pixels_rays_triangles()
            
            # get ray intersection with triangles
            points_3D = self._get_tri_ray_intersection(origins, directions, mesh_triangles)

            # # debug 
            # rays = [trimesh.load_path(np.array([origin, end])) for origin, end in zip(origins, origins+1*directions)]
            # self.scene.add_geometry(rays)
            # # debug 
            
            return pixels, points_3D
        
        @profiler.time_it
        def _get_pixels_rays_triangles(self):
            _origins, _directions, pixels = self.camera_rays 
            _tri_indices =  self.mesh.ray.intersects_first(_origins, _directions)

            # remove the -1 from the triangle indices
            mask_minus_ones = _tri_indices != -1
            tri_indices  = _tri_indices[mask_minus_ones]

            # get mesh triangles and corresponding pixels
            mesh_triangles = self.mesh_triangles[tri_indices]
            pixels = pixels[mask_minus_ones]
            origins = _origins[mask_minus_ones]
            directions = _directions[mask_minus_ones]
            
            # # debug
            # random_indices = np.random.randint(1, len(directions), size=10)
            # return pixels[random_indices], origins[random_indices], directions[random_indices], mesh_triangles[random_indices]
            # # debug

            return pixels, origins, directions, mesh_triangles
            
        @property
        def camera_rays(self): return self.scene.camera_rays()

        # @staticmethod
        # @profiler.time_it
        # @jit(nopython=True, parallel=True)
        # def _get_tri_ray_intersection(origins, directions, triangles):
        #     """
        #     Möller–Trumbore ray-triangle intersection algorithm for fast intersection testing. 
        #     """
        #     # Extract triangle vertices
        #     v0, v1, v2 = triangles[:,0,:], triangles[:,1,:], triangles[:,2,:]
        #     e1 = v1 - v0
        #     e2 = v2 - v0

        #     # Initialize arrays for the result
        #     intersection_points = np.full(origins.shape, np.nan)  # Default to NaN
            
        #     # Loop over rays in parallel using prange
        #     for i in prange(origins.shape[0]):
        #         # Calculate the cross products and other coefficients for ray-triangle intersection
        #         h = np.cross(directions[i], e2[i])
        #         a = np.dot(e1[i], h)
                
        #         # Avoid division by zero
        #         if a == 0.0:
        #             continue
                
        #         f = 1.0 / a

        #         # Calculate barycentric coordinates
        #         s = origins[i] - v0[i]
        #         u = f * np.dot(s, h)
        #         q = np.cross(s, e1[i])
        #         v = f * np.dot(directions[i], q)

        #         # Calculate intersection distance
        #         t = f * np.dot(e2[i], q)

        #         # Valid intersections: 0 <= u <= 1, 0 <= v <= 1, u + v <= 1, t >= 0
        #         if (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t >= 0):
        #             # Store the intersection point
        #             intersection_points[i] = origins[i] + directions[i] * t
            
        #     return intersection_points
               
        @staticmethod
        @profiler.time_it
        @jit(nopython=True, parallel=False)
        def _get_tri_ray_intersection(origins, directions, triangles):
            """
            Möller–Trumbore ray-triangle intersection algorithm for fast intersection testing. 
            """
            # Extract triangle vertices
            v0, v1, v2 = triangles[:,0,:], triangles[:,1,:], triangles[:,2,:]
            e1 = v1 - v0
            e2 = v2 - v0

            h = np.cross(directions, e2)
            a = np.sum(e1*h, axis=1)  
            f = 1.0/a

            # Calculate barycentric coordinates
            s = origins - v0
            u = f * np.sum(s*h, axis=1)
            q = np.cross(s, e1)
            v = f * np.sum(directions*q, axis=1)
            
            # Calculate intersection distance
            t = f * np.sum(e2*q, axis=1)
            
            # Valid intersections: 0 <= u <= 1, 0 <= v <= 1, u + v <= 1
            mask_valid = (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t >= 0)

            # Compute intersection points only for valid rays
            intersection_points = np.full(origins.shape, np.nan)  # Default to NaN
            intersection_points[mask_valid] = origins[mask_valid] + directions[mask_valid] * t[mask_valid, None]

            return intersection_points
            
    def __init__(self, filepath:str, icosphere_radius:float=0.5, icosphere_subdivisions:float=1.0, resolution:Tuple[int,int]=(480,480), fov:Tuple[float,float]=(45,45)):
        self.cameraManager = self._CameraManager(icosphere_radius, icosphere_subdivisions, resolution, fov)
        self.sceneManager = self._SceneManager(filepath)
        
        self.scene = self.sceneManager.scene
        self.camera_viewpoints = self.cameraManager.view_points
        
        # add camera to the scene
        self.scene.camera = self.cameraManager.camera

        # add the scene to 3d mapper utility
        self.mapper = self._PixelTo3DMapper(self.scene)

    def get_image(self):
        for coord in self.camera_viewpoints:
            self.scene.camera_transform = self.cameraManager.get_camera_transformation(coord)
            pixels, points_3D = self.mapper.get_map()
            # # debug -----------------------
            # axes = [trimesh.creation.axis(origin_size=0.0005, axis_length=0.005).apply_translation(loc) for loc in points_3D]
            # self.scene.add_geometry(axes)
            # self.scene.show()
            # # debug -----------------------
            
            # data = self.scene.save_image(visible=True)
            # image = np.array(Image.open(io.BytesIO(data))) 

test = IcosphereSceneGenerator("bunny.obj")
test.get_image()
profiler.generate_performance_log()