import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct as DP
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
from scipy.special import erfcinv as erfcinv
import tqdm as tqdm
import gauss_func
import time

# test comment

# np.random.multivariate_normal([0, 0], [[1,1], [1,1]] , 1)[0]
# with vars names: np.random.multivariate_normal(mean_vector_of_each_point=[0, 0], COV=[[1,1], [1,1]] , nb_maps=1)[id_map=0]
from gauss_func import gauss_func


class Point:
    """
    Object representing a point in Space
    """

    def __init__(self, label, actual_pollution_value, x, y=0):
        """
        Initalizes new point Object
        :param label: Label for point
        :param actual_pollution_value: True value of pollution at the point
        :param x: X coordinate of the point
        """
        self.label = label
        # self.actual_pollution_value = actual_pollution_value
        self.pollution_value = float('NaN')

        if actual_pollution_value < 0:
            raise Exception("Set a negative actual_pollution_value")
        self.actual_pollution_value = actual_pollution_value

        self.x = x

        self.y = y

    def __str__(self):
        return "[Point: " + str(self.label) + "| APV: " + str(self.actual_pollution_value) + "| PV: " + str(
            self.pollution_value) + "| Position: {" + str(self.x) + "," + str(self.y) + "} ]"

    def get_pollution_value(self):
        """
        Gets measured pollution value, also stores the value of the interpolated pollution value
        :return: Pollution_value (float)
        """
        return self.pollution_value

    def set_pollution_value(self, new_pollution_value):
        """
        Sets measured pollution value or to store the interpolated pollution value
        :param new_pollution_value: New pollution value to be set
        :return: Nothing
        """
        # if new_pollution_value < 0:
        #     print("SET_POLLUTION_ALERT: set pollution value to a negative value")
        self.pollution_value = new_pollution_value

    def set_actual_pollution(self, new_actual_pollution_value):
        """
        Should rarely be used changes the true value of the pollution at a point
        :param new_actual_pollution_value:
        :return:
        """

        if new_actual_pollution_value < 0:
            raise Exception("Tried to change APV to a negative actual_pollution_value")
        self.actual_pollution_value = new_actual_pollution_value

    def get_label(self):
        return self.label

    def get_x_cord(self):
        return self.x

    def get_y_cord(self):
        return self.y

    def get_position(self):
        """
        Returns Position of the point
        :return: list [x,y]
        """
        return [self.x, self.y]

    def read_pollution_value(self, standard_deviation):
        """
        Generates pollution value using actual pollution value. Currently it just assigns pollution value = actual_polution_value
        :return: Nothing
        """
        if not math.isnan(self.pollution_value):
            raise Exception("Just tried to read a pollution value from a point with a assigned pollution_value")
        self.pollution_value = self.actual_pollution_value + np.random.normal(0, standard_deviation)

    def copy(self):
        """
        Does a copy of the point
        :return: A new copied point
        """
        temp = Point(self.label, self.get_actual_pollution_value(), self.x, self.y)
        temp.set_pollution_value(self.get_pollution_value())
        return temp

    def get_actual_pollution_value(self):
        return self.actual_pollution_value


def copy_dictionary_with_points(points):
    """
    Does a deep copy of a dictionary with values being points
    :param points: A dictionary with keys being labels and values being points {label:point}
    :return: A new copied dictionary
    """
    new_dict = {}
    for key, value in points.items():
        new_dict[key] = value.copy()

    return new_dict


def distance(first_point, second_point):
    return math.sqrt(math.pow(first_point.x - second_point.x, 2) + math.pow(first_point.y - second_point.y, 2))


def create_points_with_random_pollution_2d(side_length, mean, std):
    """
    Creates a map of points with random Pollution Values using a gaussian distribution in 2D evenly starting at x = 05 every 10
    :param side_length: Number of points on each side of the square
    :param mean: Mean of gaussian distribution of pollution values
    :param std: Standard Deviation of gaussian distribution of pollution values
    :return: map of points with key being their label and the value being a point object
    """
    new_map = {}
    x = 5

    label_index = 0
    for i in range(0, side_length):
        y = 5
        for j in range(0, side_length):
            new_map[label_index] = (Point(label_index, np.random.normal(mean, std), x, y))
            label_index = label_index + 1
            y = y + 10
        x = x + 10
    return new_map


def create_points_with_spatially_correlated_pollution_2d(side_length, mean, standard_deviation, length_scale, num_maps):
    """
    Creates a map of pollution values determined by their spatial correlation
    :param side_length: Number of points on each side of the matrix
    :param mean: Mean of the gaussian distribution of pollution values
    :param std_dev: Standard Deviation of the pollution values
    :param length_scale: The length scale of the RBF kernel
    :param num_maps: Number of pollution maps used in data
    """
    # with vars names: np.random.multivariate_normal(mean_vector_of_each_point=[0, 0], COV=[[1,1], [1,1]] , nb_maps=1)[id_map=0]
    mean_vector = []
    for i in range(side_length * side_length):
        mean_vector.append(mean)
    pollution_maps = {}

    point_map = {}
    x = 5
    label_index = 0
    for i in range(0, side_length):
        y = 5
        for j in range(0, side_length):
            point_map[label_index] = Point(label_index, float('NaN'), x, y)
            label_index += 1
            y += 10
        x += 10
    for i in range(0, num_maps):
        pollution_maps[i] = copy_dictionary_with_points(point_map)

    covariance_matrix = create_covariance_matrix(point_map, length_scale, standard_deviation)
    maps_of_pollution_values = np.random.multivariate_normal(mean_vector, covariance_matrix, num_maps)

    for i in range(num_maps):
        for j in range(side_length * side_length):
            pollution_maps[i][j].set_actual_pollution(maps_of_pollution_values[i][j])

    return pollution_maps


def create_covariance_matrix(points, length_scale, standard_deviation):
    """
    Creates Covariance Matrix
    :param standard_deviation: standard deviation of pollution data
    :param points: Map of all points
    :param length_scale: Length Scale
    :return: A matrix
    """

    covariance = []
    for i in range(0, len(points)):
        covariance.append([])
        for j in range(0, len(points)):
            covariance[i].append(
                (standard_deviation ** 2) * np.exp(
                    -np.power(distance(points[i], points[j]), 2) / (2 * length_scale * length_scale)))

    return covariance


def gaussian_atmospheric_dispersion_model(source_x, source_y, side_length):
    """
    Creates a model of realistic pollution values given a source point
    :param number_of_sources: Allows normalization of pollution values, prevents a scenario where more sources equals greater maximum pollution. Dont pass anything if you dont want it normalized.
    :param source_x: x-coordinate of source point
    :param source_y: y-coordinate of source point
    :return: matrix of realistic pollution values
    """
    # SECTION 0: Definitions (normally don't modify this section)
    # view
    PLAN_VIEW = 1
    HEIGHT_SLICE = 2
    SURFACE_TIME = 3
    NO_PLOT = 4

    # wind field
    CONSTANT_WIND = 1
    FLUCTUATING_WIND = 2
    PREVAILING_WIND = 3

    # number of stacks
    ONE_STACK = 1
    TWO_STACKS = 2
    THREE_STACKS = 3

    # stability of the atmosphere
    CONSTANT_STABILITY = 1
    ANNUAL_CYCLE = 2
    stability_str = ['Very unstable', 'Moderately unstable', 'Slightly unstable', \
                     'Neutral', 'Moderately stable', 'Very stable']
    # Aerosol properties
    HUMIDIFY = 2
    DRY_AEROSOL = 1

    SODIUM_CHLORIDE = 1
    SULPHURIC_ACID = 2
    ORGANIC_ACID = 3
    AMMONIUM_NITRATE = 4
    nu = [2., 2.5, 1., 2.]
    rho_s = [2160., 1840., 1500., 1725.]
    Ms = [58.44e-3, 98e-3, 200e-3, 80e-3]
    Mw = 18e-3

    dxy = 10  # resolution of the model in both x and y directions
    dz = 10
    x = np.mgrid[5:5 + (side_length - 1) * 10 + dxy:dxy]  # solve on a 5 km domain(old comment)
    y = x  # x-grid is same as y-grid
    ###########################################################################

    # SECTION 1: Configuration
    # Variables can be changed by the user+++++++++++++++++++++++++++++++++++++
    RH = 0.90
    aerosol_type = SODIUM_CHLORIDE

    dry_size = 60e-9
    humidify = DRY_AEROSOL

    stab1 = 1  # set from 1-6
    stability_used = CONSTANT_STABILITY

    output = PLAN_VIEW
    x_slice = 26  # position (1-50) to take the slice in the x-direction
    y_slice = 1  # position (1-50) to plot concentrations vs time

    wind = CONSTANT_WIND
    stacks = ONE_STACK
    # only using one pollution source point (one stack)
    stack_x = [source_x]
    stack_y = [source_y]

    Q = [10.]  # mass emitted per unit time ::: originially 10
    H = [50.]  # [50., 50., 50.]  # stack height, m
    days = 10  # run the model for 365 days
    # --------------------------------------------------------------------------
    times = np.mgrid[1:(days) * 24 + 1:1] / 24.

    Dy = 10.
    Dz = 10.

    # SECTION 2: Act on the configuration information

    # Decide which stability profile to use
    if stability_used == CONSTANT_STABILITY:

        stability = stab1 * np.ones((days * 24, 1))
        stability_str = stability_str[stab1 - 1]
    elif stability_used == ANNUAL_CYCLE:

        stability = np.round(2.5 * np.cos(times * 2. * np.pi / (365.)) + 3.5)
        stability_str = 'Annual cycle'
    else:
        sys.exit()

    # decide what kind of run to do, plan view or y-z slice, or time series
    if output == PLAN_VIEW or output == SURFACE_TIME or output == NO_PLOT:

        C1 = np.zeros((len(x), len(y), days * 24))  # array to store data, initialised to be zero

        [x, y] = np.meshgrid(x, y)  # x and y defined at all positions on the grid
        z = np.zeros(np.shape(x))  # z is defined to be at ground level.
    elif output == HEIGHT_SLICE:
        z = np.mgrid[0:500 + dz:dz]  # z-grid

        C1 = np.zeros((len(y), len(z), days * 24))  # array to store data, initialised to be zero

        [y, z] = np.meshgrid(y, z)  # y and z defined at all positions on the grid
        x = x[x_slice] * np.ones(np.shape(y))  # x is defined to be x at x_slice
    else:
        sys.exit()

    # Set the wind based on input flags++++++++++++++++++++++++++++++++++++++++
    wind_speed = 5. * np.ones((days * 24, 1))  # m/s
    if wind == CONSTANT_WIND:
        wind_dir = 0. * np.ones((days * 24, 1))
        wind_dir_str = 'Constant wind'
    elif wind == FLUCTUATING_WIND:
        wind_dir = 360. * np.random.rand(days * 24, 1)
        wind_dir_str = 'Random wind'
    elif wind == PREVAILING_WIND:
        wind_dir = -np.sqrt(2.) * erfcinv(2. * np.random.rand(24 * days, 1)) * 40.  # norminv(rand(days.*24,1),0,40)
        # note at this point you can add on the prevailing wind direction, i.e.
        # wind_dir=wind_dir+200
        wind_dir[np.where(wind_dir >= 360.)] = \
            np.mod(wind_dir[np.where(wind_dir >= 360)], 360)
        wind_dir_str = 'Prevailing wind'
    else:
        sys.exit()
    # --------------------------------------------------------------------------

    # SECTION 3: Main loop
    # For all times...
    C1 = np.zeros((len(x), len(y), len(wind_dir)))
    for i in range(0, len(wind_dir)):
        for j in range(0, stacks):
            C = np.ones((len(x), len(y)))
            C = gauss_func(Q[j], wind_speed[i], wind_dir[i], x, y, z,
                           stack_x[j], stack_y[j], H[j], Dy, Dz, stability[i])
            C1[:, :, i] = C1[:, :, i] + C

    return np.mean(C1, axis=2) * 1e6





def create_points_using_atmospheric_model_random_locations(number_of_sources, side_length, number_of_maps,
                                                           normalized=False):
    """
     Returns a map of maps of pollution points using the Gaussian Atmospheric Dispersion Model that creates realistic
     pollution values given the number of pollution sources.

     The positions of the sources are assigned randomly
     :param normalized: Flag to set normalized to True or False
     :param number_of_sources: The number of pollution sources
     :param side_length: side length of point map square
     :param number_of_maps: number of different pollution maps used
     :return:
     """
    pollution_maps = {}

    x = 5

    for map in range(0, number_of_maps):  # loops through for each map


        pollution_values = gaussian_atmospheric_dispersion_model(np.random.randint(0, side_length * 10),
                                                                 np.random.randint(250, side_length * 10 + 250),
                                                                 side_length)  # Creates matrix of pollution using first source
        for i in range(1, number_of_sources):


            pollution_values += gaussian_atmospheric_dispersion_model(np.random.randint(0, side_length * 10),
                                                                      np.random.randint(250,
                                                                                        side_length * 10 + 250),
                                                                      side_length)  # adds additional pollution sources to pollution values
        max_poll_value = np.amax(pollution_values)
        label_index = 0
        point_map = {}
        for i in range(0, side_length):  # assigns pollution values to points
            y = 5
            for j in range(0, side_length):
                if normalized:
                    point_map[label_index] = Point(label_index, pollution_values[i][j]/max_poll_value*100, x, y)
                else:
                    point_map[label_index] = Point(label_index, pollution_values[i][j], x, y)
                label_index += 1
                y += 10
            x += 10
        pollution_maps[map] = point_map

    return pollution_maps


def interpolate_points_using_positions(known_points, wanted_point_positions, kernel=None,
                                       fixed=False, alpha=None):
    """
     Predicts points based on known data using Kriging (Gaussian Processes)
    :param known_points: list of points
    :param wanted_point_positions: list of wanted point posistions [[x1,y1], [x2,y2]]
    :param kernel:  Kernal to use in interpolation
    :param fixed:  True = no opitimization of hyperparamater, False = optimization of hyperparamter
    :param alpha: Alpha for regression ( amount of uncertainty assumed)
    :return: a list of all predicted pollution values t
    """

    # kernel = DP(1)
    # kernel = RBF(10, (1e-2, 1e2)) * C(1)

    if fixed:
        gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=10,
                                      optimizer=None, alpha=alpha)  # Instantiate a fixed Gaussian Process model
    else:
        gp = GaussianProcessRegressor(kernel,
                                      n_restarts_optimizer=10,
                                      alpha=alpha)  # Instantiate an optimized Gaussian Process model

    known_points_position_list = to_list_of_positions(known_points)

    pollution_value_list = []  # converts point list into a list of pollution values
    for point in known_points.values():
        pollution_value_list.append(point.get_pollution_value())

    gp.fit(known_points_position_list, pollution_value_list)  # Fits model to data

    #
    # prediction_list = known_points_position_list
    # prediction_list.extend(wanted_point_positions)
    # return gp.predict(prediction_list)[len(known_points):]  # predicts on new data

    return gp.predict(wanted_point_positions), gp.kernel_.length_scale


def interpolate_unknown_points(known_points, all_points, kernel=None, fixed=False, alpha=None):
    """
    Interpolate pollution values for points that are have not been measured
    :param known_points: A Dictionary of all points that have been measured {label:point}
    :param all_points: A Dictionary of all points that exist {Label: point}
    :param kernel:  Kernal to use in interpolation
    :param fixed:  True = no opitimization of hyperparamater, False = optimization of hyperparamter
    :param alpha: Alpha for regression ( amount of uncertainty assumed)
    :return: A new map with interpolated pollution values {Label : point}
    """
    unknown_positions = []
    unknown_labels = []
    for i in range(0, len(
            all_points)):  # Creates a list of posistions of points that have not been measured and a list of their respective labels
        if not (i in known_points):
            unknown_positions.append(all_points.get(i).get_position())
            unknown_labels.append(i)

    interpolated_pollution_values, length_scale = interpolate_points_using_positions(known_points,
                                                                                     unknown_positions,
                                                                                     kernel,
                                                                                     fixed,
                                                                                     alpha=alpha)  # interpolates the pollution values for the posistions where we have not measured pollution values

    interpolated_map = copy_dictionary_with_points(known_points)  # creates a copy of the known_points dictionary

    for i in range(0, len(
            unknown_labels)):  # adds missing points and their interpolated pollution values into the new dictionary
        interpolated_map[unknown_labels[i]] = all_points.get(unknown_labels[i]).copy()
        interpolated_map[unknown_labels[i]].set_pollution_value(interpolated_pollution_values[i])

    return (interpolated_map, length_scale)


def interpolate_unknown_points_of_a_map_of_maps_of_points(known_points, all_points, kernel=None, fixed=False,
                                                          alpha=None):
    """

    :param alpha: Alpha for regression ( amount of uncertainty assumed)
    :param known_points: A map of maps of all points that have been measured
    :param all_points:  A  map of maps of all points that exist
    :param kernel:  Kernal to use in interpolation
    :param fixed:  True = no opitimization of hyperparamater, False = optimization of hyperparamter
    :return: A tuple of (A new map of maps of interpolated values, length of kernel)
    """

    interpolated_maps = {}
    for label in all_points.keys():
        interpolated_maps[label] = interpolate_unknown_points(known_points[label], all_points[label], kernel, fixed,
                                                              alpha)

    return interpolated_maps


def pick_uniform_random_points_on_map_of_maps(points, pick_number, standard_deviation):
    """
        Picks a number of points from a list of points using a uniform random distribution for each map in a map
        :param points: Map of all the points {label: point}
        :param pick_number: Number of points to pick
        :param mean: mean pollution value
        :param standard_deviation: standard deviation of pollution data
        :return: A new map with the picked points {label: point}
        """

    new_map = {}
    for label in points.keys():
        new_map[label] = pick_uniform_random_points(points[label], pick_number, standard_deviation)

    return new_map


def pick_uniform_random_points(points, pick_number, standard_deviation):
    """
    Picks a number of points from a list of points using a uniform random distribution
    :param points: Map of all the points {label: point}
    :param pick_number: Number of points to pick
    :param standard_deviation: standard deviation of measurment of pollution data
    :return: A new map with the picked points {label: point}
    """
    random = np.random.default_rng()

    random_picks = []

    while len(random_picks) < pick_number:
        random_num = random.integers(0, len(points))  # picks uniform random number
        if not random_num in random_picks:  # Checks to make sure number hasnt been picked yet
            random_picks.append(random_num)
    new_map = {}
    for i in random_picks:  # assigns and copies picked points into a new dictionary
        new_map[i] = Point(i, points.get(i).get_actual_pollution_value(), points.get(i).get_x_cord(),
                           points.get(i).get_y_cord())
        new_map.get(i).read_pollution_value(standard_deviation)

    return new_map


def to_list_of_positions(points):
    """
    Converts a dictionary of points into a list of posistions [[x1,y1],[x2,y2]]
    :param points: A dictionary of points {label : point}
    :return: A list of positions of the points [[x1,y1],[x2,y2]]
    """
    points_position_list = []  # converts point list into a list of [x,y] values
    for point in points.values():
        points_position_list.append(point.get_position())

    return points_position_list


def root_mean_square_error(points):
    """
    Finds the RMSE of the interpolated pollution values
    param points: A dictionary of points {label : point}
    There is a manual flag in the code to switch from regular rmse to relative rmse
    :return: The RMSE of the pollution values found through interpolation
    """

    relative = True  # allows manual change between relative rmse and  rmse

    if not relative:  # if just using rmse
        sum = 0
        for point in points.values():  # does rmse formula
            sum += pow(np.abs(point.get_pollution_value()) - np.abs(point.get_actual_pollution_value()), 2)
        rmse = math.sqrt(sum / len(points))
        return rmse
    else:  # if we want to use relative rmse
        sum = 0
        mean = 0
        max_value = 0
        for point in points.values():  # does rmse formula
            sum += pow(point.get_pollution_value() - point.get_actual_pollution_value(), 2)
            # mean += abs(point.get_actual_pollution_value())
            if max_value < point.get_actual_pollution_value():
                max_value = point.get_actual_pollution_value()
        rmse = math.sqrt(sum / len(points))
        # mean /= len(points.values())
        return rmse / max_value  # divides by max to normalize the rmse


def average_rmse_of_maps(maps_of_points):
    """

    :param maps_of_points:
    :return: Returns double of average rmse of maps
    """
    num_of_maps = len(maps_of_points)
    sum = 0
    for label in maps_of_points.keys():
        sum += root_mean_square_error(maps_of_points[label][0])
    return sum / num_of_maps


def plot_numbers(x_axis, y_axis, x_label, y_label, x_log_scale=False, x_axis_2=None, y_axis_2=None, ):
    """
    Plots Numbers on a graph
    :param y_label: Label for Y-axis of graph
    :param x_label: Label for X-axis of graph
    :param x_axis, x_axis_2: X-value for the graph in list form
    :param y_axis, y_axis_2: Y-values for the graph in list form
    :return:
    """
    if x_axis_2 == None or y_axis_2 == None:
        plt.plot(x_axis, y_axis, "ro")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if x_log_scale:
            plt.xscale("log")
    else:
        plt.plot(x_axis, y_axis, "ro", x_axis_2, y_axis_2, "go")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if x_log_scale:
            plt.xscale("log")

    plt.show()


def plot_bar_graph(x_axis, y_axis, x_label, y_label):
    """
    Plots a bar graph for RMSE values with respect to the number of sources used in the pollution map
    :param x_axis: values for the x-axis of the graph
    :param y_axis: values for the y-axis of the graph
    :param x_label: variable on x-axis
    :param y_label: variable on y-axis
    :return: a plotted bar graph
    """
    plt.bar(x_axis, y_axis)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plot_varied_std_deviations(length_scale_list, std_deviations, x_label, y_label):
    """
    Plots RMSE vs Length Scale of pollution points with various standard deviations
    :param length_scale_list: list of used length scales on x-axis
    :param std_deviations: list of various standard deviations
    :param x_label: label for x-axis of graph
    :param y_label: label for y-axis of graph
    :return:
    """

    plt.plot(length_scale_list, std_deviations[0], "ro",
             length_scale_list, std_deviations[1], "go", length_scale_list, std_deviations[2], "bo")
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.show()


def list_of_length_scales(bottom_bound, top_bound, steps):
    """
    Creates an array of used length scales during the graphing process
    :param bottom_bound: lower bound of length scale
    :param top_bound: upper bound of length scale
    :param steps: difference between each adjacent length scale
    :return: a list of various length scales
    """
    length_scales = []
    for i in range(bottom_bound, top_bound, steps):
        length_scales.append(i)
    return length_scales


def plot_numbers_3d_and_save(x1, y1, z1, x2, y2, z2, filename="Rotating Graph.gif"):
    """
    Scatterplot in 3d and saves rotating gif to file
    :param x1:
    :param y1:
    :param z1:
    :param x2:
    :param y2:
    :param z2:
    :param filename: File name ending in .gif
    :return:
    """
    fig = plt.figure()
    sub = fig.add_subplot(1, 1, 1, projection="3d")
    sub.scatter(x1, y1, z1, marker="o", edgecolor="r", facecolor="r")
    sub.scatter(x2, y2, z2, marker="^", edgecolor="g", facecolor="g")

    def rotate(angle):
        sub.view_init(azim=angle)

    rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 2), interval=100)
    # mywriter = animation.FFMpegWriter(fps=60)
    # rot_animation.save("rotation.mp4",dpi = 80, writer= mywriter)
    print("Starting Save")
    rot_animation.save(filename, dpi=80, writer='imagemagick')
    print("Finished Save")


def graph_pollution_using_heat_map(points, title, side_length):
    """
    Creates HeatMap graph of pollution in 2d
    :param points: Map of points
    :param title: Title of graph
    :param side_length: Side length of points box
    :return:
    """
    plt.figure()
    plt.ion()

    y, x = np.mgrid[slice(5, (side_length - 1) * 10 + 10, 10),
                    slice(5, (side_length - 1) * 10 + 10, 10)]  # create grid for pcolormesh
    pollution = []
    max_pollution = 0

    for i in range(0, side_length):  # initalizes empty matrix
        pollution.append([])
        for j in range(0, side_length):
            pollution[i].append(0)

    x_pos = 0
    y_pos = 0
    for label in range(0, len(points.keys())):  # fill matrix with correct pollution data
        pollution[x_pos][y_pos] = points[label].get_pollution_value()
        if max_pollution < points[label].get_pollution_value():  # finds max pollution value for coloring
            max_pollution = points[label].get_pollution_value()
        y_pos += 1
        if x_pos == side_length:
            raise Exception("Error, This should never be true in graph_pollution_using_heat_map")

        if y_pos == side_length:
            y_pos = 0
            x_pos += 1

    plt.pcolormesh(x, y, pollution, cmap='jet')  # graphing functions
    plt.clim((0, max_pollution))
    plt.title(title)
    plt.xlabel('x (metres)')
    plt.ylabel('y (metres)')
    cb1 = plt.colorbar()
    cb1.set_label("Pollutants")  # old label = '$\mu$ g m$^{-3}$'
    plt.show()


def graph_error_based_on_different_number_sources(side_length, max_number_of_sources, number_of_maps, num_picked_points,
                                                  error_of_measurment,
                                                  normalized_pollution_values=False):
    """
    Method to test how RMSE varies when changing the number of pollution sources on map
    :param side_length: side length of pollution square map
    :param max_number_of_sources: maximum number of sources used
    :param number_of_maps: number of maps used in calculations
    :param num_picked_points: number of known pollution values on graph
    :param error_of_measurment: standard deviation of error in measurement
    :param normalized_pollution_values: decides whether to normalize the pollution values or not
    :return:
    """
    rmse_data = []
    for current_num_sources in range(1, max_number_of_sources + 1): # loops through number of sources to be used
        points = create_points_using_atmospheric_model_random_locations(current_num_sources, side_length,
                                                                        number_of_maps,
                                                                        normalized_pollution_values) # creation of pollution points
        picked_points = pick_uniform_random_points_on_map_of_maps(points, num_picked_points,
                                                                  standard_deviation=error_of_measurment) # selecting known points
        interpolated_points = interpolate_unknown_points_of_a_map_of_maps_of_points(picked_points, points,
                                                                                    RBF(np.random.randint(1e-05,
                                                                                                          100)), False,
                                                                                    alpha=.1) # interpolating points with specific parameters
        rmse_data.append(average_rmse_of_maps(interpolated_points)) # adding average rmse of interpolated points maps to a list of rmse data

        print("Source number:" + str(current_num_sources) + " Done")
        graph_heatmap_best_interpolation(points, interpolated_points, side_length, current_num_sources) # graphs the best interpolation

    plot_bar_graph(range(1, max_number_of_sources + 1), rmse_data, x_label="number of sources", y_label="RMSE") # plots a bar graph of
    #                                                                                                            RMSE vs number of sources


def graph_heatmap_best_interpolation(points, interpolated_points, side_length, number_of_sources):
    """
    Graphs a pollution heatmap using the best possible interpolation of poitns
    :param points: map of pollution points
    :param interpolated_points: map of interpolated points
    :param side_length: side length of pollution square map
    :param number_of_sources: number of pollution sources on graph
    :return: a heatmap of pollution values
    """
    true_pollution_points = pick_uniform_random_points_on_map_of_maps(points, side_length ** 2, 0) # initializing true values for pollution
    #                                                                                                 map
    num_of_maps = len(points)
    rmse_list = []
    min_rmse = math.inf
    min_label = None
    max_label = None
    max_rmse = 0

    # finding minimum and maximum rmse from list of RMSE values from maps of interpolated points
    for label in interpolated_points.keys():

        rmse = root_mean_square_error(interpolated_points[label][0])
        rmse_list.append(rmse)
        if rmse < min_rmse:
            min_rmse = rmse
            min_label = label

        if rmse > max_rmse:
            max_rmse = rmse
            max_label = label

    graph_pollution_using_heat_map(true_pollution_points[min_label],
                                   "best true values" + ", Number of Sources = " + str(
                                       number_of_sources), side_length) # graphs pollution heatmap of the best true pollution values
    graph_pollution_using_heat_map(interpolated_points[min_label][0], "best interpolated values| RMSE = " + str(
        truncate(min_rmse, 3)) + ", Number of Sources = " + str(number_of_sources), side_length) # graphs pollution heatmap of best
    #                                                                                              interpolated pollution values

    graph_pollution_using_heat_map(true_pollution_points[max_label],
                                   "worst true values" + ", Number of Sources = " + str(
                                       number_of_sources), side_length) # graphs heatmap of worst "true" pollution values
    graph_pollution_using_heat_map(interpolated_points[max_label][0], "worst interpolated values| RMSE = " + str(
        truncate(max_rmse, 3)) + ", Number of Sources = " + str(number_of_sources), side_length) # graphs heatmap of worst
    #                                                                                              interpolated pollutino values


def truncate(number, digits) -> float:
    """
    Truncates a number to a certain number of digits
    :param number: number to be truncated
    :param digits: number of decimal places to truncate to
    :return: a specifically truncated number
    """
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper


def experiment_test_all_alphas(lower_alpha, higher_alpha, side_length, std_of_measurments, max_number_of_sources,
                               number_of_maps, num_picked_points, normalized_pollution_values):
    """

    :param lower_alpha: Lower bound of alpha
    :param higher_alpha: Higher bound of alpha
    :param side_length: Number of points on one side of the point square
    :param std_of_measurments: Standard deviation of pollution measurments
    :param max_number_of_sources: Max number of sources
    :param number_of_maps: Number of simulations for each parameter
    :param num_picked_points: Number of points that will be measured before interpolation
    :param normalized_pollution_values:  True or False
    :return:
    """
    rmse_data = {}
    tree = 0
    for current_alpha in np.arange(lower_alpha, higher_alpha, .1):
        current_alpha = truncate(current_alpha, 3)
        rmse_data_for_certain_alpha = {}
        for current_num_sources in range(1, max_number_of_sources + 1):
            points = create_points_using_atmospheric_model_random_locations(current_num_sources, side_length,
                                                                            number_of_maps,
                                                                            normalized=normalized_pollution_values)
            picked_points = pick_uniform_random_points_on_map_of_maps(points, num_picked_points,
                                                                      standard_deviation=std_of_measurments)
            interpolated_points = interpolate_unknown_points_of_a_map_of_maps_of_points(picked_points, points,
                                                                                        RBF(np.random.randint(1e-05,
                                                                                                              100)),
                                                                                        False,
                                                                                        alpha=current_alpha)
            rmse_data_for_certain_alpha[current_num_sources] = average_rmse_of_maps(interpolated_points)
            print("Source number:" + str(current_num_sources) + " Done")

        rmse_data[current_alpha] = rmse_data_for_certain_alpha
        print("Alpha:" + str(current_alpha) + " Done")

    min_rmse = math.inf
    min_alpha = None
    avg_rmse_map_of_all_alphas = {}

    for alpha, rmse_list in rmse_data.items():
        sum = 0
        for value in rmse_list.values():
            sum += value
        avg_rmse = sum / len(rmse_list.values())
        if min_rmse > avg_rmse:
            min_rmse = avg_rmse
            min_alpha = alpha

    for num_sources in range(1, max_number_of_sources + 1):
        sum = 0
        for alpha in np.arange(lower_alpha, higher_alpha, .1):
            alpha = truncate(alpha, 3)
            sum += rmse_data[alpha][num_sources]
        mean = sum / len(np.arange(lower_alpha, higher_alpha, .1))
        avg_rmse_map_of_all_alphas[num_sources] = mean

    x_cord = range(1, max_number_of_sources + 1)
    y1_cord = put_y_values_in_right_order(rmse_data[min_alpha])
    y2_cord = put_y_values_in_right_order(avg_rmse_map_of_all_alphas)

    # plt.scatter(x_cord, y1_cord, "ro", x_cord, y2_cord, "go")
    plt.plot(x_cord, y1_cord, "ro-", x_cord, y2_cord, "go-")
    plt.xlabel("Number of Sources")
    plt.ylabel("RSME")
    plt.title("Best Alpha = " + str(min_alpha))
    plt.show()
    print("done")


def put_y_values_in_right_order(map):
    """
    Rearranges the y values of the map
    :param map: map to be rearranged
    :return:
    """
    key_list = []
    y_cord = []
    for key in map.keys():
        key_list.append(key)
    key_list.sort()
    for key in key_list:
        y_cord.append(map[key])
    return y_cord



"""
Testing Methods
"""
# list_of_std_deviations = [1, 5, 10]
# run_experiment_with_varied_standard_deviations(bottom_bound=10, top_bound=100, steps= 5, side_length= 10, mean =150, std_of_pollution= 10,
#                                                std_deviation_values_of_measurment= list_of_std_deviations, pick_number= 20, num_maps= 100)


# run_experiment_with_various_length_scales_log(.000001, 1000000, 10, 100, 20, 2)
# run_experiment_with_various_length_scales_linear(bottom_bound=10, top_bound=100, step =5,
#                                                  side_length=10, mean=100, pick_number=20,
#                                                  number_of_maps=100, standard_deviation=10)


# see_what_its_doing_2d(length_scale=40,cheating= True,pollution_mean= 150, pollution_std= 10, pick_number= 50)
# see_what_its_doing_2d_comparison(10,True)


#  Playing around with gaussian disperssion stuff

# side_length = 40
#
# points = create_points_using_atmospheric_model([200], [500], side_length, 1)
# b = pick_uniform_random_points_on_map_of_maps(points, side_length ** 2, 0)
# graph_pollution_using_heat_map(b[0], "Graph", side_length=side_length)

# graph_error_based_on_different_number_sources(number_of_maps=5, max_number_of_sources=10, side_length=40,
#                                               num_picked_points=150, error_of_measurment=5,
#                                               normalized_pollution_values=True)

experiment_test_all_alphas(lower_alpha=.1, higher_alpha=2, side_length=40, std_of_measurments=5,
                           max_number_of_sources=5, number_of_maps=20, num_picked_points=100,
                           normalized_pollution_values=True)

# experiment_test_all_alphas(lower_alpha=.1, higher_alpha=.5, side_length=40, std_of_measurments=5,
#                            max_number_of_sources=5, number_of_maps=10, num_picked_points=20,
#                            normalized_pollution_values=True)
