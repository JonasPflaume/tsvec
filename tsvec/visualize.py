import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp


def visualize_trajectory(
    init_state,
    pcloud,
    sdf_fn=None,
    sdf_bounds=None,
    sdf_resolution=64,
    number=0,
    interactive=False,
    save_image=True,
    output_path=None,
):
    if output_path is None:
        output_path = f"{number}.png"

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="TSVec Ergodic Trajectory Planning", width=1200, height=900)
    render_option = vis.get_render_option()
    render_option.background_color = np.array([1, 1, 1]) 
    render_option.light_on = True
    render_option.point_size = 12.0 
    render_option.line_width = 15.0 


    pcloud_o3d = o3d.geometry.PointCloud()
    pcloud_vertices = np.array(pcloud.vertices)
    pcloud_colors = np.array(pcloud.colors)
    enhanced_colors = np.copy(pcloud_colors)
    brightness_factor = 1.5
    enhanced_colors = np.clip(enhanced_colors * brightness_factor, 0, 1)
    dark_mask = np.sum(enhanced_colors, axis=1) < 0.3
    enhanced_colors[dark_mask] = np.array([0.1, 0.1, 0.8])
    pcloud_o3d.points = o3d.utility.Vector3dVector(pcloud_vertices)
    pcloud_o3d.colors = o3d.utility.Vector3dVector(enhanced_colors)
    vis.add_geometry(pcloud_o3d)


    if sdf_fn is not None and sdf_bounds is not None:
        from skimage import measure


        x = np.linspace(sdf_bounds[0][0], sdf_bounds[0][1], sdf_resolution)
        y = np.linspace(sdf_bounds[1][0], sdf_bounds[1][1], sdf_resolution)
        z = np.linspace(sdf_bounds[2][0], sdf_bounds[2][1], sdf_resolution)
        grid = np.stack(np.meshgrid(x, y, z, indexing='ij'), -1)
        grid_flat = grid.reshape(-1, 3)
        sdf_vals = np.array(jax.vmap(sdf_fn)(grid_flat)).reshape(sdf_resolution, sdf_resolution, sdf_resolution)
        verts, faces, normals, _ = measure.marching_cubes(sdf_vals, level=0.0, spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0]))

        center = np.mean(verts, axis=0)
        verts = (verts - center) * 0.99 + center
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts + np.array([sdf_bounds[0][0], sdf_bounds[1][0], sdf_bounds[2][0]]))
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.7, 0.7, 0.7])
        vis.add_geometry(mesh)

    if init_state is not None:
        state = init_state
        positions = state.pos_particles
        z_axes = state.rot_particles.apply(jnp.array([0.0, 0.0, 1.0]))

        for i in range(positions.shape[0]):
            trajectory_points = np.array(positions[i])
            N = trajectory_points.shape[0]
            jet_cmap = plt.get_cmap("jet")
            traj_colors = jet_cmap(np.linspace(0.8, 0.99, N))[:, :3]
            traj_colors = traj_colors * 0.9 

            for j in range(N-1):
                start = trajectory_points[j]
                end = trajectory_points[j+1]
                color = traj_colors[j]
                direction = end - start
                height = np.linalg.norm(direction)
                if height < 1e-6:
                    continue
                cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.006, height=height)
                cylinder.paint_uniform_color(color.tolist())
                default_direction = np.array([0, 0, 1])
                direction_normalized = direction / height
                axis = np.cross(default_direction, direction_normalized)
                angle = np.arccos(np.clip(np.dot(default_direction, direction_normalized), -1, 1))
                if np.linalg.norm(axis) > 1e-6:
                    axis = axis / np.linalg.norm(axis)
                    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
                    cylinder.rotate(rotation_matrix, center=[0, 0, 0])
                center = (start + end) / 2
                cylinder.translate(center)
                vis.add_geometry(cylinder)

            for j in range(N):
                pos = trajectory_points[j]
                z_dir = np.array(z_axes[i, j])
                if np.linalg.norm(z_dir) > 1e-6:
                    arrow_z = o3d.geometry.TriangleMesh.create_arrow(
                        cylinder_radius=0.0033,
                        cone_radius=0.006,
                        cylinder_height=0.05,
                        cone_height=0.02
                    )
                    arrow_z.paint_uniform_color(traj_colors[j].tolist())
                    default_direction = np.array([0, 0, 1])
                    z_dir_normalized = z_dir / np.linalg.norm(z_dir)
                    axis = np.cross(default_direction, z_dir_normalized)
                    angle = np.arccos(np.clip(np.dot(default_direction, z_dir_normalized), -1, 1))
                    if np.linalg.norm(axis) > 1e-6:
                        axis = axis / np.linalg.norm(axis)
                        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
                        arrow_z.rotate(rotation_matrix, center=[0, 0, 0])
                    arrow_z.translate(pos)
                    vis.add_geometry(arrow_z)

    view_control = vis.get_view_control()
    view_control.set_front([1.0, -0.4, -0.8])
    view_control.set_lookat([0.5, 0.0, 0.5])
    view_control.set_up([0, 0, 1])
    view_control.set_zoom(0.7)
    render_option.light_on = True

    if interactive:
        def save_current_view(vis):
            vis.capture_screen_image(str(output_path), do_render=True)
            print(f"Saved current view to: {output_path}")
            return False

        if save_image:
            vis.register_key_callback(ord("S"), save_current_view)
            vis.register_key_callback(ord("s"), save_current_view)
            print("Interactive Open3D window opened. Rotate/zoom freely; press S to save the current view.")
        else:
            print("Interactive Open3D window opened. Rotate/zoom freely.")
        vis.run()
    else:
        def capture_and_exit(vis):
            if save_image:
                vis.capture_screen_image(str(output_path), do_render=True)
            vis.close()
            return False

        vis.register_animation_callback(capture_and_exit)
        vis.run()

    vis.destroy_window()
