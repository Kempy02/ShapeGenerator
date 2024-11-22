# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from geomdl import NURBS
import cadquery as cq
from cadquery import exporters
import math

# General Parameters
n_curves = 5  # Number of curves

# Curve-related Parameters
degree = 3
order = degree + 1  # Order is degree + 1
knot_c = 1
resolution = 100  # The resolution of the NURBS curves

# Geometry-related Parameters
start_x = 0
start_y = 0
amplitude0 = 20  # Initial amplitude
period1 = 10     # Initial period (for curve1 and curve2)
period2 = 10     # Period 2 (for curve3 and curve4)
period3 = 10     # Period 3 (for curve5)
cap_height = 5   # Cap height for curve0
cap_length = 5   # Cap length for curve0
thickness = 1    # Cross-section thickness
revolve_offset = 1  # Offset from the center for revolve
period_values = [period1, period2, period3]  # Period values for each curve

# Control Point Parameters
offset_factor_x0 = 0         # Offset in x-direction
offset_factor_y0 = 1         # Offset in y-direction
mid_factor_x = 0             # Midpoint x displacement factor
mid_factor_y = 0             # Midpoint y displacement factor
min_y_positions = [0, 0, 0]  # Minimum y positions for curves
true_mid = 1                 # Whether to fix midpoint to start/end y-position (1=True, 0=False)
rel_mid = 1 - true_mid       # 
cp1_weight = 1               # Weight for control point 1
cp3_weight = 1               # Weight for control point 3
weights = [1, cp1_weight, 1, cp3_weight, 1]  # Control point weights for curves

def validateParameters(n_curves, periodValues, min_y_positions):
    """
    Validate the input parameters to ensure they are acceptable.

    Parameters:
        n_curves (int): Total number of curves.
        period_values (list): List of periods for each curve.
        min_y_positions (list): List of minimum y positions for descending curves.

    Returns:
        tuple: Validated or adjusted period_values and min_y_positions lists.
    """
    # Number of periods (assuming each period consists of 2 curves)
    n_periods = int(np.ceil(n_curves / 2))

    # Number of descending curves
    n_descending_curves = int(np.floor(n_curves / 2))

    # Validate lengths of min_y_positions and period_values
    if len(min_y_positions) < n_descending_curves + 1:
        raise ValueError(f'Length of min_y_positions must be at least {n_descending_curves + 1}')
    
    if len(periodValues) < n_periods:
        raise ValueError(f'Length of period_values must be at least {n_periods}')
    
    if len(min_y_positions) > n_descending_curves + 1:
        # If there are extra min_y_positions, truncate and give a warning
        min_y_positions = min_y_positions[:n_descending_curves + 1]
        print(f'Warning: Extra elements in min_y_positions. Using the first {n_descending_curves + 1} elements.')
    else:
        # If the length is acceptable
        min_y_positions = min_y_positions
    
    if len(periodValues) > n_periods:
        # If there are extra period_values, truncate and give a warning
        period_values = periodValues[:n_periods]
        print(f'Warning: Extra elements in period_values. Using the first {n_periods} elements.')
    else:
        # If the length is acceptable
        periodValues = periodValues

    return periodValues, min_y_positions

def computeDerivedParameters(n_curves, start_y, amplitude0, period_values, min_y_positions):
    """
    Compute x_increments and y_positions based on the input parameters.

    Parameters:
        n_curves (int): Total number of curves.
        start_y (float): The starting y-position for the curves.
        amplitude0 (float): The initial amplitude for the ascending curves.
        period_values (list): The list of period values for each curve.
        min_y_positions (list): The list of minimum y positions for descending curves.

    Returns:
        tuple: x_increments (list), y_positions (list)
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
        n_control_points (int): Number of control points in the curve.
        degree (int): Degree of the NURBS curve.
        knot_c (float): Knot spacing multiplier (default is 1).

    Returns:
        list: A clamped knot vector.
    """
    n = n_control_points - 1
    order = degree + 1
    internal_knots = n - order + 1
    if internal_knots >= 0:
        knot_vector = [0] * order + [i * knot_c for i in range(1, internal_knots + 1)] + [(internal_knots + 1) * knot_c] * order
    else:
        print(f'Degree: {degree}')
        print(f'n_control_points: {n_control_points}')
        raise ValueError('The number of control points must be greater than or equal to the degree.')
    return knot_vector

def generateCapCurve(start_x, start_y, cap_length, cap_height, degree, resolution, weights):
    """
    Generates the cap curve (curve0).

    Parameters:
        start_x (float): Starting x-coordinate.
        start_y (float): Starting y-coordinate.
        cap_length (float): Length of the cap.
        cap_height (float): Height of the cap.
        degree (int): Degree of the NURBS curve (usually 2 for the cap).
        resolution (int): Number of points to evaluate on the curve.
        weights (list): Weights for the control points.

    Returns:
        tuple: (curve_points, control_points, control_points_idx, curve_points_idx)
    """
    # Initialize lists to collect all curve points and control points and respective index's
    all_control_points = []
    all_curve_points = []

    # Define control points for the cap
    control_points = [
        [start_x, start_y],
        [start_x + cap_length, start_y],
        [start_x + cap_length, start_y + cap_height]
    ]
    all_control_points.append(control_points)

    # Create index vector for control points
    control_points_idx = np.arange(1, len(control_points) + 1)

    # Generate knot vector
    knot_vector = createKnotVector(len(control_points), degree)

    # Create a NURBS curve instance
    curve = NURBS.Curve()
    curve.degree = degree
    curve.ctrlpts = control_points  # 2D control points
    curve.weights = [1, 1, 1]
    curve.knotvector = knot_vector
    curve.delta = 1 / resolution  # Set resolution

    # Evaluate the curve
    curve.evaluate()
    curve_points = np.array(curve.evalpts)
    all_curve_points.append(curve_points)

    # Create index vector for curve points
    curve_points_idx = np.arange(1, len(curve_points) + 1)

    end_x = all_control_points[-1][0]
    end_y = all_control_points[-1][1]

    return all_curve_points, all_control_points, control_points_idx, curve_points_idx, end_x, end_y

def generateCSCurves(n_curves, degree, resolution, current_start_x, current_start_y,
                             x_increments, y_positions, weights,
                             offset_factor_x0, offset_factor_y0,
                             mid_factor_x, mid_factor_y, true_mid, all_curve_points, all_control_points):
    """
    Generates the sequential NURBS curves.

    Parameters:
        n_curves (int): Number of curves to generate.
        degree (int): Degree of the NURBS curve.
        resolution (int): Number of points to evaluate on the curve.
        start_x (float): Starting x-coordinate.
        start_y (float): Starting y-coordinate.
        x_increments (list): Increments in x-direction for each curve.
        y_positions (list): y-positions for each curve.
        weights (list): Weights for the control points.
        offset_factor_x0 (float): Offset factor in x-direction.
        offset_factor_y0 (float): Offset factor in y-direction.
        mid_factor_x (float): Midpoint x displacement factor.
        mid_factor_y (float): Midpoint y displacement factor.
        true_mid (int): Whether to fix midpoint to start/end y-position.

    Returns:
        tuple: (all_curve_points, all_control_points, control_points_idx, curve_points_idx)
    """
    # all_control_points = []
    # all_curve_points = []
    control_points_idx = []
    curve_points_idx = []

    # current_start_x = start_x# + cap_length
    # current_start_y = start_y# + cap_height
    start_x = cap_length
    # start_y = current_start_y

    for i in range(n_curves):

        # Compute start and end point of the current curve
        start_y = cap_height + y_positions[i]
        end_y = cap_height + y_positions[i + 1]

        # Compute polarity based on the direction of the curve
        polarity = np.sign(end_y - start_y)
        
        # Compute start and end point in x-direction
        dx = x_increments[i]
        if i < 1:
            end_x = start_x + dx/2
        else:
            end_x = start_x + dx
        
        # Compute mid point
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        
        # Offset factors
        offset_factor_x = offset_factor_x0
        offset_factor_y = offset_factor_y0 * polarity
        
        # Define control points
        if i < 1:
            cp0_x = (start_x)
            cp0_y = start_y
        else:
            cp0_x = start_x
            cp0_y = start_y

        if i < 1:
            cp2_x = cp0_x
        else:
            cp2_x = mid_x + ((end_x - mid_x) * mid_factor_x) * polarity
        cp2_y = mid_y + ((end_x - mid_x) * mid_factor_y) * polarity

        if i < 1:
            cp1_y = cp0_y
            cp1_x = cp0_x
        else:
            cp1_y = mid_y * true_mid + cp2_y * rel_mid - abs(start_y - mid_y) * offset_factor_y
            cp1_x = cp2_x + (end_x - mid_x) * offset_factor_x

        cp3_x = cp2_x - (end_x - mid_x) * offset_factor_x
        cp3_y = mid_y * true_mid + cp2_y * rel_mid + abs(start_y - mid_y) * offset_factor_y

        if i < n_curves - 1:
            cp4_x = end_x
        else:
            cp4_x = end_x - thickness
        cp4_y = end_y

        # Control points in homogeneous coordinates
        control_points = [
            [cp0_x, cp0_y],
            [cp1_x, cp1_y],
            [cp2_x, cp2_y],
            [cp3_x, cp3_y],
            [cp4_x, cp4_y]
        ]

        # Append control points
        all_control_points.append(control_points)

        # Create index arrays for control points
        cp_idx = np.arange(1, len(control_points) + 1)
        control_points_idx.append(cp_idx)

        # Generate knot vector
        knot_vector = createKnotVector(len(control_points), degree)

        # print(control_points)

        # Create NURBS curve instance
        curve = NURBS.Curve()
        curve.degree = degree
        curve.ctrlpts = [ [pt[0], pt[1]] for pt in control_points ]
        curve.weights = weights
        curve.knotvector = knot_vector
        curve.delta = 1 / resolution

        # Evaluate the curve
        curve.evaluate()
        curve_points = np.array(curve.evalpts)
        if i < n_curves - 1:
            all_curve_points.append(curve_points[:-1])
        else:
            all_curve_points.append(curve_points)

        # Create index arrays for curve points
        curve_idx = np.arange(1, len(curve_points) + 1)
        if i < n_curves - 1:
            curve_points_idx.append(curve_idx[:-1])
        else:
            curve_points_idx.append(curve_idx)

        # Update start positions for next curve
        start_x = end_x
        start_y = end_y

    return all_curve_points, all_control_points, control_points_idx, curve_points_idx

def aggregateCurveData(curve_points_list, control_points_list):
    """
    Aggregates all curve points and control points into single arrays.

    Parameters:
        curve_points_list (list): List of numpy arrays containing curve points.
        control_points_list (list): List of lists containing control points.

    Returns:
        tuple: (all_curve_points, all_control_points)
    """
    # Combine all curve points into a single array
    new_curve_points = np.vstack(curve_points_list)
    # new_curve_points = np.concatenate(curve_points_list)

    # Combine all control points into a single array
    new_control_points = np.vstack(control_points_list)

    return new_curve_points, new_control_points

def processOuterPoints(new_curve_points):
    """
    Process outer points by inverting and shifting them to start at (0, 0).

    Parameters:
        all_points (list): List of all points.

    Returns:
        numpy.ndarray: Processed outer points.
    """

   
    outer_points = np.vstack(new_curve_points)
    outer_points = np.array(outer_points)

    # print(outer_points)

    # Extract max x-coordinate
    max_x = max(point[0] for point in outer_points)

    # Invert outer points
    outer_points = outer_points[::-1]

    # Move curve to begin at (0, 0)
    outer_points = np.array([((x - max_x), y) for x, y in outer_points])

    return outer_points

def computeNormals(outer_points):
    """
    Compute tangents and normals along the curve.

    Parameters:
        all_curve_points (numpy.ndarray): Array of curve points.

    Returns:
        tuple: (tangents, normals)
    """
    tangents = []
    normals = []

    for i in range(len(outer_points) - 1):
        # Get the current point and the next point
        x1, y1 = outer_points[i]
        x2, y2 = outer_points[i + 1]

        # Compute the difference in coordinates
        dx = x2 - x1
        dy = y2 - y1

        # Compute the length of the segment
        length = math.hypot(dx, dy) or 1e-8  # Avoid division by zero

        # Compute the tangent vector (normalized)
        tx = dx / length
        ty = dy / length
        tangents.append((tx, ty))

        # Compute the normal vector by rotating the tangent by 90 degrees
        nx = -ty
        ny = tx
        normals.append((nx, ny))

    # Append the last normal vector
    normals.append(normals[-1])

    return tangents, normals

def generateVTControlPoints(all_control_points, control_points_idx):
    """
    Generate the VT control points by processing the control points.

    Parameters:
        all_control_points (list): List of all control points.
        control_points_idx (list): List of control points indices.

    Returns:
        numpy.ndarray: Array of VT control points.
    """
    # Replace all even indexes with 0
    vt_cp_idx = [np.where(np.arange(len(idx)) % 2 == 1, 0, idx) for idx in control_points_idx]

    # Arrange all_control_points to match vt_cp_idx
    vt_control_points = []
    for i in range(len(all_control_points)):
        cp_array = all_control_points[i]
        vt_cp_idx_array = vt_cp_idx[i]
        vt_control_points.append([cp_array[j] for j in range(len(cp_array)) if vt_cp_idx_array[j] != 0])

    # Invert vt_control_points to match the order of the curve points
    vt_control_points = [cp[::-1] for cp in vt_control_points[::-1]]

    # Extract max x-coordinate
    max_x = max(point[0] for cp in vt_control_points for point in cp)

    # Move control points to begin at (0,0)
    vt_control_points = [[(x - max_x, y) for x, y in cp] for cp in vt_control_points]

    return vt_control_points

def calculateThicknessProfiles(outer_points, control_points_list, all_thicknesses, vt_control_points):
    """
    Calculate the thickness at each point along the outer curve.

    Parameters:
        outer_points (numpy.ndarray): Array of outer curve points.
        control_points_list (list): List of control points for each curve.
        thicknesseses (list): List of thickness values at control points for each curve.

    Returns:
        list: Thickness value at each point along the outer curve.
    """

    # Create 1D array for vt_control_points
    vt_control_points_array = np.vstack(vt_control_points)

    # Invert all_thicknesses to match
    all_thicknesses = [t[::-1] for t in all_thicknesses[::-1]]

    # Find the indices of the closest curve points to each control point
    cp_in_curve_idx = []

    for cp in vt_control_points_array:
        x_cp, y_cp = cp
        # Compute distances to all curve points
        distances = [math.hypot(x_cp - x_curve, y_cp - y_curve) for x_curve, y_curve in outer_points]
        # Find index of the closest curve point
        idx = distances.index(min(distances))
        # print(min(distances))
        cp_in_curve_idx.append(idx)

    print(len(outer_points))
    print(vt_control_points_array)
    print(cp_in_curve_idx)

    # Error check
    if cp_in_curve_idx[0] != 0 or cp_in_curve_idx[-1] != len(outer_points) - 1:
        print(all_thicknesses)
        raise ValueError("First value of cp_in_curve_idx must be 0 and last value must be equal to the length of outer_points minus one.")


    # Arrange cp_in_curve_idx into segmented arrays to match the structure of vt_control_points
    segmented_cp_in_curve_idx = []
    start_idx = 0

    for segment in vt_control_points:
        segment_length = len(segment)
        end_idx = start_idx + segment_length
        segmented_cp_in_curve_idx.append(cp_in_curve_idx[start_idx:end_idx])
        start_idx = end_idx

    # Remove duplicate points within each segment
    unique_segmented_cp_in_curve_idx = []
    for segment in segmented_cp_in_curve_idx:
        unique_segment = []
        for idx in segment:
            if idx not in unique_segment:
                unique_segment.append(idx)
        unique_segmented_cp_in_curve_idx.append(unique_segment)
    segmented_cp_in_curve_idx = unique_segmented_cp_in_curve_idx

    # Calculate thicknesses along the curve
    point_thicknesses = []
    for i in range(len(control_points_list)):
        thicknesses = all_thicknesses[i]
        cp_indices = segmented_cp_in_curve_idx[i]

        # Ensure the number of thickness values matches the number of control point indices
        if len(thicknesses) != len(cp_indices):
            raise ValueError(f"Thickness values and control point indices mismatch in curve {i+1}.")

        for j in range(len(cp_indices) - 1):
            start_idx = cp_indices[j]
            end_idx = cp_indices[j + 1]
            n_points = end_idx - start_idx
            if n_points == 0:
                continue  # Avoid division by zero
            start_thickness = thicknesses[j]
            end_thickness = thicknesses[j + 1]
            for k in range(n_points):
                t = k / n_points
                thickness = start_thickness + t * (end_thickness - start_thickness)
                point_thicknesses.append(thickness)

    # Append the last thickness
    point_thicknesses.append(thickness)

    return point_thicknesses

def generateInnerProfile(outer_points, normals, point_thicknesses):
    """
    Generate the inner profile by offsetting outer points along normals.

    Parameters:
        outer_points (numpy.ndarray): Array of outer curve points.
        normals (list): List of normal vectors at each point.
        point_thicknesses (list): Thickness at each point.

    Returns:
        list: Inner profile points.
    """
    inner_points = []

    for i in range(len(outer_points)):
        x, y = outer_points[i]
        nx, ny = normals[i]
        thickness = point_thicknesses[i]
        offset_x = x + nx * thickness
        offset_y = y + ny * thickness
        inner_points.append((offset_x, offset_y))

    return inner_points

def createCrossSection(outer_points, inner_points):
    """
    Create a closed cross-section by combining outer and inner profiles.

    Parameters:
        outer_points (numpy.ndarray): Array of outer curve points.
        inner_points (list): List of inner profile points.

    Returns:
        list: Cross-section points forming a closed loop.
    """
    # Reverse outer points for correct orientation
    outer_points_reversed = outer_points[::-1]
    outer_points_list = [tuple(pt) for pt in outer_points_reversed]
    
    # Combine inner and outer points
    cross_section_points = inner_points + outer_points_list

    # Close the profile
    cross_section_points.append(cross_section_points[0])

    return cross_section_points

def generate3DModel(cross_section_points, revolve_offset):
    """
    Generate a 3D model by revolving the cross-section.

    Parameters:
        cross_section_points (list): List of points forming the cross-section.
        revolve_offset (float): Offset from the center for the revolve axis.

    Returns:
        cadquery.Workplane: The 3D model object.
    """
    # Create the cross-section profile
    cross_section = cq.Workplane("XY").polyline(cross_section_points).close()

    # Revolve the profile around the Y-axis
    model = (
        cross_section
        .revolve(360, (revolve_offset, 0, 0), (revolve_offset, 1, 0))
        .translate((-revolve_offset, 0, 0))
    )

    return model

def createCap(outer_max_y, revolve_offset, cap_thickness):
    """
    Create a cap and position it on top of the model.

    Parameters:
        outer_max_y (float): The maximum y-coordinate of the outer points.
        revolve_offset (float): Offset from the center for the revolve axis.
        cap_thickness (float): Thickness of the cap.

    Returns:
        cadquery.Workplane: The cap object.
    """
    cap = (
        cq.Workplane('XZ')
        .circle(revolve_offset)
        .extrude(cap_thickness)
        .translate((0, outer_max_y, 0))
    )
    return cap

def createBase(base_params):
    """
    Create the base of the model.

    Parameters:
        base_params (dict): Dictionary containing base parameters.

    Returns:
        cadquery.Workplane: The base object.
    """
    # Unpack base parameters
    base_radius = base_params['base_radius']
    base_plate_height = base_params['base_plate_height']
    base_internal_radius = base_params['base_internal_radius']
    base_internal_height = base_params['base_internal_height']
    wall_thickness = base_params['wall_thickness']
    base_extension = base_params['base_extension']
    screw_radius = base_params['screw_radius']
    desired_side_length = base_params['desired_side_length']

    # Create base
    base = (
        cq.Workplane("XZ")
        .circle(base_radius)
        .extrude(base_plate_height)
        # Add screw holes
        .faces(">Y")
        .rect(desired_side_length, desired_side_length, forConstruction=True)
        .vertices()
        .circle(screw_radius)
        .cutThruAll()
        # Hollow internal walls
        .faces(">Y")
        .workplane()
        .circle(base_internal_radius - wall_thickness)
        .extrude(-base_internal_height, combine='cut')
    )

    return base


def main():
    # [Parameters are already defined at the top]
    
    # Validate parameters
    validateParameters(n_curves, period_values, min_y_positions)
    
    # Compute derived parameters
    x_increments, y_positions = computeDerivedParameters(n_curves, start_y, amplitude0, period_values, min_y_positions)
    
    # Generate cap curve
    cap_degree = 2  # Degree for the cap curve
    cap_weights = [1, 3, 1]  # Adjust weights if needed
    
    all_curve_points, all_control_points, cap_cp_idx, cap_curve_idx, end_x, end_y = generateCapCurve(
        start_x, start_y, cap_length, cap_height, cap_degree, resolution, cap_weights
    )
    
    # Generate sequential curves
    all_curve_points, all_control_points, control_points_idx, curve_points_idx = generateCSCurves(
        n_curves, degree, resolution, end_x, end_y,
        x_increments, y_positions, weights,
        offset_factor_x0, offset_factor_y0,
        mid_factor_x, mid_factor_y, true_mid, 
        all_curve_points, all_control_points
    )
    
    # Append cap data to the lists
    # all_curve_points = [cap_curve_points[:-1]] + all_curve_points  # Exclude last point to avoid duplication
    # all_control_points = [cap_control_points] + all_control_points
    control_points_idx = [cap_cp_idx] + control_points_idx
    curve_points_idx = [cap_curve_idx] + curve_points_idx
    
    # Aggregate curve data
    new_curve_points = aggregateCurveData(all_curve_points, all_control_points)

    # Process Outer Points data
    outer_points = processOuterPoints(new_curve_points)
    
    # Compute normals
    normals = computeNormals(outer_points)


    vt_control_points = generateVTControlPoints(all_control_points, control_points_idx)
    
    # Define thickness profiles for each curve
    # thicknesses = [
    #     [1, 1],             # For cap curve (curve0)
    #     [1, 0.5, 1],        # For curve1
    #     [1, 0.5, 1],        # For curve2
    #     [1, 0.5, 1],        # For curve3
    #     [1, 0.5, 1],        # For curve4
    #     [1, 0.5, 1]         # For curve5
    # ]

    # Define thicknesses at each control point for each curve segment
    thicknesses0 = [1, 1]
    thicknesses1 = [1, 0.5, 1]
    thicknesses2 = [1, 0.5, 1]
    thicknesses3 = [1, 0.5, 1]
    thicknesses4 = [1, 0.5, 1]
    thicknesses5 = [1, 0.5, 1]

    all_thicknesses = []
    for i in range(n_curves + 1):
        all_thicknesses.append(eval(f'thicknesses{i}'))
    
    # Calculate thickness profiles
    point_thicknesses = calculateThicknessProfiles(outer_points, all_control_points, all_thicknesses, vt_control_points)
    
    # Generate inner profile
    inner_points = generateInnerProfile(outer_points, normals, point_thicknesses)
    
    # Create cross-section
    cross_section_points = createCrossSection(outer_points, inner_points)
    
    # Adjust cross-section points to start at (0, 0)
    first_x, first_y = cross_section_points[0]
    cross_section_points_shifted = [((x - first_x), y) for x, y in cross_section_points]
    
    # Generate 3D model
    model = generate3DModel(cross_section_points_shifted, revolve_offset)
    
    # Get maximum y-coordinate from outer points
    outer_max_y = np.max(outer_points[:, 1]) - first_y  # Adjust for shift
    
    # Determine cap thickness (e.g., use the thickness at the center point)
    cap_thickness = point_thicknesses[len(point_thicknesses) // 2]
    
    # Create cap
    cap = createCap(outer_max_y, revolve_offset, cap_thickness)
    
    # Add cap to the model
    model_with_cap = model + cap
    
    # Define base parameters (adjust values as needed)
    base_params = {
        'base_radius': 50,
        'base_plate_height': 2,
        'base_internal_radius': 40,
        'base_internal_height': 20,
        'wall_thickness': 3,
        'base_extension': 10,
        'screw_radius': 1.5,
        'desired_side_length': 60,
    }
    
    # Create base
    base = createBase(base_params)
    
    # Combine model with base
    final_model = model_with_cap.union(base)
    
    # Export final model
    exportFilename = "final_model.stl"
    profileName = final_model
    profileName.export(exportFilename)
    print(f"Model exported as {exportFilename}")

if __name__ == '__main__':
    main()