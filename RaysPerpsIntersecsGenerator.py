'''
Code whritten and published by Leonid O. Shabaev in January 2025
'''

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Font parameters
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.size'] = 10

# Pixelation parameters and number of rays
pixel_axis_size = 2                         # Number of pixels along the axes from zero
num_rays = 36                               # Number of rays

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Generate points on a grid
def generate_points_on_grid(point_type: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(-pixel_axis_size / 2, pixel_axis_size)
    y = np.arange(-pixel_axis_size / 2, pixel_axis_size)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()]).T
    return points, np.full(len(points), point_type)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Rays generation
def generate_rays_around_points(points):
    max_distance = np.linalg.norm(points, axis=1).max()
    angles = np.linspace(0, 2 * np.pi, num=num_rays, endpoint=False)
    rays = []

    for idx, point in enumerate(tqdm(points, desc="Ray generation", total=len(points))):
        for angle_idx, angle in enumerate(angles):
            ray_x = [point[0], point[0] + 2 * max_distance * np.cos(angle)]
            ray_y = [point[1], point[1] + 2 * max_distance * np.sin(angle)]
            ray_id = f"{idx}_{angle_idx}"
            
            rays.append({
                "point_index": idx,
                "ray_index": angle_idx,
                "ray": (ray_x, ray_y),
                "point_type": "first_type",
                "perp_index": angle_idx,
                "ray_id": ray_id
            })
    return rays


# Perpendiculars generation
def find_perpendicular_lines(point, ray):
    ray_x1, ray_y1 = ray["ray"][0][0], ray["ray"][1][0]
    ray_x2, ray_y2 = ray["ray"][0][1], ray["ray"][1][1]

    # If the ray is vertical
    if np.isclose(ray_x1, ray_x2):
        x_perpendicular = ray_x1
        y_perpendicular = point["position"][1]
    # If the ray is horizontal
    elif np.isclose(ray_y1, ray_y2):
        x_perpendicular = point["position"][0]
        y_perpendicular = ray_y1
    else:
    # For inclined rays
        k_ray = (ray_y2 - ray_y1) / (ray_x2 - ray_x1)
        k_perpendicular = -1 / k_ray

        b_ray = ray_y1 - k_ray * ray_x1
        b_perpendicular = point["position"][1] - k_perpendicular * point["position"][0]

        x_perpendicular = (b_perpendicular - b_ray) / (k_ray - k_perpendicular)
        y_perpendicular = k_ray * x_perpendicular + b_ray

    return {
        "perpendicular": [[point["position"][0], x_perpendicular], [point["position"][1], y_perpendicular]],
        "ray_index": ray["ray_index"],
        "point_index": ray["point_index"],
        "ray_id": ray["ray_id"],
        "intersection_point": [x_perpendicular, y_perpendicular],
        "second_type_point_index": point["id"],
        "perp_index": ray["perp_index"]
    }


# Determining intersection points
def find_intersection_points(second_type_points, rays):
    intersections = []
    for second_type_point in tqdm(second_type_points, desc="Determining intersection points", total=len(second_type_points)):
        for ray in rays:
            intersection = find_perpendicular_lines(second_type_point, ray)
            if intersection is not None:
                intersection["belongs_to_perpendicular"] = True
                intersection["ray_index"] = ray["ray_index"]
                intersection["perp_index"] = ray["perp_index"]
                intersection["second_type_point_index"] = second_type_point["id"]
                intersection["ray_id"] = ray["ray_id"]
                intersections.append(intersection)
    return intersections


# Drawing perpendiculars
def drawing_perpendiculars(points, rays, batch_size=num_rays):
    perpendiculars_batches = []
    for start in range(0, len(points), batch_size):
        end = start + batch_size
        batch_points = points[start:end]
        batch_perpendiculars = []
        for point in tqdm(batch_points, desc="Second type points generation", total=len(batch_points)):
            for ray in rays:
                result = find_perpendicular_lines(point, ray)
                if result is not None:
                    batch_perpendiculars.append(result)
        perpendiculars_batches.extend(batch_perpendiculars)
    return perpendiculars_batches

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Create a map of the distribution of rays, perpendiculars and intersection points
def generate_fig():
    fig, ax = plt.subplots(figsize=(8, 8))


    # Generate points for first and second types
    Second_Type_Points, _ = generate_points_on_grid("second_type")
    First_Type_Points, _ = generate_points_on_grid("first_type")

    # Convert second type points to dictionaries
    second_type_points_dict = [{"id": idx, "position": pos} for idx, pos in enumerate(Second_Type_Points)]

    # Generate rays
    Rays = generate_rays_around_points(First_Type_Points)

    # Generate perpendiculars
    Perpendiculars = drawing_perpendiculars(second_type_points_dict, Rays)

    # Find intersection points
    Intersections = find_intersection_points(second_type_points_dict, Rays)

    # Ray drawing
    for Ray in Rays:
        ax.plot(Ray["ray"][0], Ray["ray"][1], color='yellow', alpha=1, zorder=2)

    # Drawing perpendiculars
    for Perpendicular in Perpendiculars:
        ax.plot(Perpendicular["perpendicular"][0], Perpendicular["perpendicular"][1], color='green', alpha=0.3, zorder=3)
    
    # Drawing the second type of points
    ax.scatter(Second_Type_Points[:, 0], Second_Type_Points[:, 1], s=10, c='red', alpha=1, zorder=4)
    
    # Drawing intersection points
    intersection_points = np.array([inter["intersection_point"] for inter in Intersections])  # Точки пересечения уже в kpc
    ax.scatter(intersection_points[:, 0], intersection_points[:, 1], s=1, c='orange', alpha=1, zorder=5)

    # Figure title
    ax.set_title('Rays, Perpendiculars and Intersection Points')
    # Ticks properties
    tick_step = 1
    ticks = np.arange(-pixel_axis_size, pixel_axis_size + tick_step, tick_step)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # Axes names
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')

    plt.show()

    return fig, ax

generate_fig()
