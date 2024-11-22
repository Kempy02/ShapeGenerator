import numpy as np
import matplotlib.pyplot as plt
from geomdl import NURBS
import cadquery as cq
from cadquery import exporters
import cq_editor as cqe
import shapely.geometry as sg
from shapely.geometry import LineString, Polygon
import math as math

# Feature Flags (Global at the top of the document)
PLOT_CURVES = True
GENERATE_CAD_FILES = True
VERBOSE = False

# User-Variable Parameters
# Curves Parameters
n_curves = 5  # Number of NURBS curves to generate
degree = 3  # Degree of NURBS curve
order = 4  # Order = degree + 1
knot_c = 1  # Knot vector parameter
no_control_points = 5  # Number of control points
resolution = 100  # Resolution for NURBS curve (number of points to evaluate)
amplitude0 = 20  # Amplitude of the curve
period0 = 10  # Initial period for curve generation
period_values = [10, 10, 10]  # Periods for each curve segment
start_x = 0  # Starting x-coordinate
start_y = 0  # Starting y-coordinate

# Cap/Curve0 Parameters
cap_height = 5  # Height for the cap
cap_length = 5  # Length of the cap
curve0_cp1_weight = 3  # Weight at cp1 for curve0

# Control Point Parameters
offset_factor_x0 = 0  # Offset factor in x for control points
offset_factor_y0 = 1  # Offset factor in y for control points
mid_factor_x = 0  # Mid-point displacement factor x
mid_factor_y = 0  # Mid-point displacement factor y
min_y_positions = [0, 0, 0]  # Minimum y-positions for each period
true_mid = 1  # Controls if cp1/cp3 are fixed or relative to mid-point
cp1_weight = 3  # Weight at cp1
cp3_weight = 3  # Weight at cp3


# Thickness Parameters
variable_thickness_period0 = [1, 1, 1]  # Control points for variable thickness at period 0
variable_thickness_period1 = [1, 0.5, 1]  # Control points for variable thickness at period 1
variable_thickness_period2 = [1, 0.5, 1]  # Control points for variable thickness at period 2
variable_thickness_period3 = [1, 0.5, 1]  # Control points for variable thickness at period 3
variable_thickness_period4 = [1, 0.5, 1]  # Control points for variable thickness at period 4
variable_thickness_period5 = [1, 0.5, 1]  # Control points for variable thickness at period 5

# Base Parameters
base_extension = 10  # Extension for the base
base_internal_radius = 15  # Internal radius of the base
base_plate_height = 2  # Height of the base plate
base_plate_width = 2  # Width of the base plate
wall_thickness = 3  # Thickness of the walls

screw_diameter = 3  # Radius of the screws [cm]
screw_head_diameter = 5.5  # Radius of the screw head [cm]
screw_tolerance = 1.0  # Tolerance around the screw [cm]

clamp_side_length = 30  # Length of the clamp side
clamp_depth = 30  # Depth of the clamp
jimstron_clamp_plate_width = 8  # Jimstron clamp plate width
jimstron_clamp_max_distance = 56  # Max distance for the clamp

# Held (Fixed) Parameters

# Fixed curve parameters
cp0_weight = 1,  # Weight at cp0
cp2_weight = 1,  # Weight at cp2
cp4_weight = 1,  # Weight at cp4
curve0_cp0_weight = 1  # Weight at cp0 for curve0
curve0_cp2_weight = 1  # Weight at cp2 for curve0
degree0 = 2  # Degree for the initial cap curve
knot_c0 = 1  # Knot vector parameter for the initial cap curve

# Fixed Thickness Parameters
thickness_points = 3  # Thickness of the cross-section
thickness_sections = n_curves  # Number of thickness sections

# Fixed Base Parameters
screw_radius = screw_diameter / 2  # Radius of the screws
screw_head_radius = screw_head_diameter / 2  # Radius of the screw head

# Fixed CAD Parameters
# revolve_offset = thickness  # Offset from the center for the revolve

# def initialize_parameters():
    
    

#     return 

def validate_parameters(n_curves, period_values, min_y_positions):
    """
    Validate the input parameters to ensure they are acceptable.
    Parameters:
        n_curves: Total number of curves.
        period_values: List of periods for each curve.
        min_y_positions: List of minimum y positions for descending curves.
    Returns:
        Updated period_values and min_y_positions after validation.
    """
    # Number of periods (assuming each period consists of 2 curves)
    n_periods = int(np.ceil(n_curves / 2))
    
    # Number of descending curves
    n_descending_curves = int(np.floor(n_curves / 2))

    # Validate lengths of min_y_positions and period_values
    if len(min_y_positions) < n_descending_curves + 1:
        raise ValueError(f"Length of min_y_positions must be at least {n_descending_curves + 1}.")
    elif len(period_values) < n_periods:
        raise ValueError(f"Length of period_values must be at least {n_periods}.")
    elif len(min_y_positions) > n_descending_curves + 1:
        # If there are extra min_y_positions, truncate and issue a warning
        min_y_positions = min_y_positions[:n_descending_curves + 1]
        print(f"Warning: Extra min_y_positions provided. Truncated to {n_descending_curves + 1} elements.")
    elif len(period_values) > n_periods:
        # If there are extra period_values, truncate and issue a warning
        period_values = period_values[:n_periods]
        print(f"Warning: Extra period_values provided. Truncated to {n_periods} elements.")

    return period_values, min_y_positions

def compute_curve_parameters(n_curves, start_y, amplitude0, period_values, min_y_positions):
    """
    Compute x_increments and y_positions based on the input parameters.
    Parameters:
        n_curves: Total number of curves.
        start_y: The starting y-position for the curves.
        amplitude0: The initial amplitude for the ascending curves.
        period_values: The list of period values for each curve.
        min_y_positions: The list of minimum y positions for descending curves.
    Returns:
        x_increments: List of x increments for each curve.
        y_positions: List of y positions for each curve.
    """
    x_increments = [0] * n_curves
    y_positions = [0] * (n_curves + 1)
    y_positions[0] = start_y

    p = 0  # Period counter
    for i in range(n_curves):
        if i % 2 == 0:  # Ascending curve
            amplitude = amplitude0 - min_y_positions[p]
            y_positions[i + 1] = y_positions[i] + amplitude
            x_increments[i] = period_values[p] / 2
        else:  # Descending curve
            y_positions[i + 1] = min_y_positions[p + 1]
            x_increments[i] = period_values[p] / 2
            p += 1

    return x_increments, y_positions

def createKnotVector(n_control_points, degree, knot_c=1):
    """
    Generates a clamped knot vector for the NURBS curve.
    Parameters:
        n_control_points: Number of control points in the curve.
        degree: Degree of the NURBS curve.
        knot_c: Knot spacing multiplier (default is 1).
    Returns:
        knot_vector: A list representing the clamped knot vector.
    """
    n = n_control_points - 1
    order = degree + 1
    internal_knots = n - order + 1
    if (n + order - 2*order) >= 0:
        knot_vector = [0] * order + [(i * knot_c) for i in range(1, internal_knots+1)] + [internal_knots + 1] * order
    else:
        raise ValueError('Number of control points must be greater than or equal to the degree.')
    return knot_vector

 
def calculate_control_points(i, params):
    
    control_points = []
    # ...
    return control_points


def evaluate_nurbs_curve(control_points, params):
    # Use geomdl to create and evaluate the NURBS curve
    from geomdl import NURBS
    curve = NURBS.Curve()
    curve.degree = params.degree
    curve.ctrlpts = control_points
    curve.weights = params.weights
    curve.knotvector = generate_knot_vector(params)
    curve.delta = 1 / params.resolution
    curve.evaluate()
    curve_points = np.array(curve.evalpts)
    return curve_points


def calculate_thicknesses(params, all_control_points, all_curve_points):
    point_thicknesses = []
    # Implement the logic to calculate thickness at each point
    # ...
    return point_thicknesses


def generate_profiles(params, outer_points, point_thicknesses, normals):
    inner_points = []
    for i in range(len(outer_points)):
        xGet the outer point coordinates
        x_x, y = outer_points[i]
        
        #Get the normal vector at this point
        nx, ny = normals[i]
        
        # Get the thickness at this point
        thickness = point_thicknesses[i]
        
        # Offset the point along the normal vector
        offset_x = x_x + nx * thickness
        offset_y = y + ny * thickness
        inner_points.append((offset_x, offset_y))
    return inner_points


def generate_cad_model(params, inner_points, outer_points):
    if not params.generate_cad_files:
        return
    # Use CadQuery to create the 3D model
    # ...
    # Export the model
    # ...


def plot_curves(params, all_curve_points, all_control_points):
    if not params.plot_curves:
        return
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for i, curve_points in enumerate(all_curve_points):
        plt.plot(curve_points[:, 0], curve_points[:, 1], label=f'Curve {i+1}')
        control_points = all_control_points[i]
        cp_x = [pt[0] for pt in control_points]
        cp_y = [pt[1] for pt_pt in control_points]
        plt.plot(cp_x, cp_y, 'o--', label=f'Control Polygon {i+1}')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Sequential NURBS Curves')
    plt.grid(True)
    plt.show()


def main():
    
    # Generate curves
    # Generate the cap curve (curve0)
    all_curve_points, all_control_points, control_points_idx, curve_points_idx = generate_curves(
    n_curves=0, degree=2, order=3, knot_c=1, resolution=100,
    start_x=0, start_y=0, control_points_input=curve0_control_points, weights=weights0, generate_cap=True
)
    
    # Calculate thicknesses
    point_thicknesses = calculate_thicknesses(params, all_control_points, all_curve_points)
    
    # Calculate normals
    normals = calculate_normals(all_curve_points)
    
    # Generate profiles
    outer_points = np.vstack(all_curve_points)
    inner_points = generate_profiles(params, outer_points, point_thicknesses, normals)
    
    # Generate CAD model
    generate_cad_model(params, inner_points, outer_points)
    
    # Plot curves
    plot_curves(params, all_curve_points, all_control_points)


if __name__ == '__main__':
    main()