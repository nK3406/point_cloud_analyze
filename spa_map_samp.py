import sys
import time
import pyzed.sl as sl
import ogl_viewer.viewer as gl

CREATE_MESH = True

def main(resolution = 0.025, save_texture = False, period = 1.5):
    print("Running Spatial Mapping sample ... Press 'q' to quit")

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080 # Use HD720 video mode
    init_params.coordinate_units = sl.UNIT.METER         # Set coordinate units
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP  # OpenGL coordinates

    # If applicable, use the SVO given as parameter
    # Otherwise use ZED live stream
    if len(sys.argv) == 2:
        filepath = sys.argv[1]
        print("Using SVO file: {0}".format(filepath))
        init_params.set_from_svo_file(filepath)

    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    # Get camera parameters
    camera_parameters = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam

    if CREATE_MESH:
        pymesh = sl.Mesh()              # Current incremental mesh
    else:
        pymesh = sl.FusedPointCloud()   # Current incremental FusedPointCloud
    image = sl.Mat()                    # Left image from camera
    pose = sl.Pose()                    # Camera pose tracking data

    viewer = gl.GLViewer()
    viewer.init(camera_parameters, pymesh, CREATE_MESH)

    spatial_mapping_parameters = sl.SpatialMappingParameters()
    tracking_state = sl.POSITIONAL_TRACKING_STATE.OFF
    mapping_state = sl.SPATIAL_MAPPING_STATE.NOT_ENABLED
    mapping_activated = True
    last_call = time.time()             # Timestamp of last mesh request

    # Enable positional tracking
    err = zed.enable_positional_tracking()
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        exit()

    # Set runtime parameters
    runtime = sl.RuntimeParameters()

    while viewer.is_available():
        # Grab an image, a RuntimeParameters object must be given to grab()
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            # Update pose data (used for projection of the mesh over the current image)
            tracking_state = zed.get_position(pose)

            if mapping_activated:
                mapping_state = zed.get_spatial_mapping_state()
                # Compute elapsed time since the last call of Camera.request_spatial_map_async()
                duration = time.time() - last_call  
                # Ask for a mesh update if 500ms elapsed since last request
                if(duration > .5 and viewer.chunks_updated()):
                    zed.request_spatial_map_async()
                    last_call = time.time()
                
                if zed.get_spatial_map_request_status_async() == sl.ERROR_CODE.SUCCESS and round(time.time(),1)%period == 0:
                    zed.retrieve_spatial_map_async(pymesh)
                    viewer.update_chunks()
                
            change_state = viewer.update_view(image, pose.pose_data(), tracking_state, mapping_state)

            if change_state:
                if not mapping_activated:
                    init_pose = sl.Transform()
                    zed.reset_positional_tracking(init_pose)

                    # Configure spatial mapping parameters
                    spatial_mapping_parameters.resolution_meter = resolution # sl.SpatialMappingParameters().get_resolution_preset(sl.MAPPING_RESOLUTION.MEDIUM)
                    spatial_mapping_parameters.use_chunk_only = True
                    spatial_mapping_parameters.save_texture = save_texture       # Set to True to apply texture over the created mesh
                    spatial_mapping_parameters.range_meter = 2.0
                    if CREATE_MESH:
                        spatial_mapping_parameters.map_type = sl.SPATIAL_MAP_TYPE.MESH
                    else:
                        spatial_mapping_parameters.map_type = sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD

                    # Enable spatial mapping
                    zed.enable_spatial_mapping(spatial_mapping_parameters)

                    # Clear previous mesh data
                    pymesh.clear()
                    viewer.clear_current_mesh()

                    # Start timer
                    last_call = time.time()

                    mapping_activated = True
                else:
                    # Extract whole mesh
                    zed.extract_whole_spatial_map(pymesh)

                    if CREATE_MESH:
                        filter_params = sl.MeshFilterParameters()
                        filter_params.set(sl.MESH_FILTER.MEDIUM) 
                        # Filter the extracted mesh
                        pymesh.filter(filter_params, True)
                        viewer.clear_current_mesh()

                        # If textures have been saved during spatial mapping, apply them to the mesh
                        if(spatial_mapping_parameters.save_texture):
                            print("Save texture set to : {}".format(spatial_mapping_parameters.save_texture))
                            pymesh.apply_texture(sl.MESH_TEXTURE_FORMAT.RGBA)

                    # Save mesh as an obj file
                    filepath = "zed_data/mesh_gen.obj"
                    status = pymesh.save(filepath)
                    if status:
                        print("Mesh saved under " + filepath)
                    else:
                        print("Failed to save the mesh under " + filepath)
                    
                    mapping_state = sl.SPATIAL_MAPPING_STATE.NOT_ENABLED
                    mapping_activated = False
    
    image.free(memory_type=sl.MEM.CPU)
    pymesh.clear()
    # Disable modules and close camera
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    zed.close()
    
if __name__ == "__main__":
    main()