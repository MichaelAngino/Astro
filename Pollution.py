import numpy as np
import math
import matplotlib.pyplot as plt
# import sklearn as sk
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


# test comment

class Point:
    """
    Object representing a point in Space
    """

    def __init__(self, label, actual_pollution_value, x):
        """
        Initalizes new point Object
        :param label: Label for point
        :param actual_pollution_value: True value of pollution at the point
        :param x: X coordinate of the point
        """
        self.label = label
        # self.actual_pollution_value = actual_pollution_value
        self.pollution_value = float('NaN')

        self.actual_pollution_value = actual_pollution_value

        self.x = x

        self.y = 0

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
        self.pollution_value = new_pollution_value

    def get_label(self):
        return self.label

    def get_x_cord(self):
        return self.x

    def get_position(self):
        """
        Returns Position of the point
        :return: list [x,y]
        """
        return [self.x, self.y]

    def read_pollution_value(self):
        """
        Generates pollution value using actual pollution value. Currently it just assigns pollution value = actual_polution_value
        :return: Nothing
        """
        if not math.isnan(self.pollution_value):
            raise Exception("Just tried to read a pollution value from a point with a assigned pollution_value")
        self.pollution_value = self.actual_pollution_value

    def copy(self):
        """
        Does a copy of the point
        :return: A new copied point
        """
        temp = Point(self.label, self.get_actual_pollution_value(), self.x)
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


def create_points_with_random_pollution(length, mean, std):
    """
    Creates a map of points with random Pollution Values using a gaussian distribution in 1D evenly starting at x = 05 every 10
    :param length: Wanted length of list
    :param mean: Mean of gaussian distribution of pollution values
    :param std: Standard Deviation of gaussian distribution of pollution values
    :return: map of points with key being their label and the value being a point object
    """
    new_map = {}
    x = 5
    for i in range(0, length):
        new_map[i] = (Point(i, np.random.normal(mean, std), x))
        x = x + 10
    return new_map


def interpolate_points_using_positions(known_points, wanted_point_positions):
    """
    Predicts points based on known data using Kriging (Gaussian Processes)
    :param known_points: list of points
    :param wanted_point_positions: list of wanted point posistions [[x1,y1], [x2,y2]]
    :return: a list of all predicted pollution values
    """

    gp = GaussianProcessRegressor(C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)))  # Instantiate a Gaussian Process model

    known_points_position_list = to_list_of_positions(known_points)

    pollution_value_list = []  # converts point list into a list of pollution values
    for point in known_points.values():
        pollution_value_list.append(point.get_pollution_value())

    gp.fit(known_points_position_list, pollution_value_list)  # Fits model to data

    prediction_list = known_points_position_list
    prediction_list.extend(wanted_point_positions)
    return gp.predict(prediction_list)[len(known_points):]  # predicts on new data


def pick_uniform_random_points(points, pick_number):
    """
    Picks a number of points from a list of points using a uniform random distribution
    :param points: Map of all the points {label: point}
    :param pick_number: Number of points to pick
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
        new_map[i] = Point(i, points.get(i).get_actual_pollution_value(), points.get(i).get_x_cord())
        new_map.get(i).read_pollution_value()

    return new_map


def pick_poisson_random_points(points, pick_number, lam):
    """
    Picks random points using poisson distribution
    :param points: All possible points
    :param pick_number: Number of points to pick
    :param lam: Lambda variable for the Poisson Function (typically a number)
    :return: A dictionary of points
    """
    random = np.random.default_rng()

    random_picks = []

    while len(random_picks) < pick_number:
        random_num = random.poisson(lam)  # picks poisson random number
        if not random_num in random_picks and random_num > 0 and random_num < len(
                points):  # Checks to make sure number hasnt been picked yet and is within range
            random_picks.append(random_num)
    new_map = {}
    for i in random_picks:  # assigns and copies picked points into a new dictionary
        new_map[i] = Point(i, points.get(i).get_actual_pollution_value(), points.get(i).get_x_cord())
        new_map.get(i).read_pollution_value()

    return new_map


def interpolate_unknown_points(known_points, all_points):
    """
    Interpolate pollution values for points that are have not been measured
    :param known_points: A Dictionary of all points that have been measured {label:point}
    :param all_points: A Dictionary of all points that exist {Label: point}
    :return: A new map with interpolated pollution values {Label : point}
    """
    unknown_positions = []
    unknown_labels = []
    for i in range(0, len(
            all_points)):  # Creates a list of posistions of points that have not been measured and a list of their respective labels
        if not (i in known_points):
            unknown_positions.append(all_points.get(i).get_position())
            unknown_labels.append(i)

    interpolated_pollution_values = interpolate_points_using_positions(known_points,
                                                                       unknown_positions)  # interpolates the pollution values for the posistions where we have not measured pollution values

    interpolated_map = copy_dictionary_with_points(known_points)  # creates a copy of the known_points dictionary

    for i in range(0, len(
            unknown_labels)):  # adds missing points and their interpolated pollution values into the new dictionary
        interpolated_map[unknown_labels[i]] = all_points.get(unknown_labels[i]).copy()
        interpolated_map[unknown_labels[i]].set_pollution_value(interpolated_pollution_values[i])

    return interpolated_map


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
    :return: The RMSE of the pollution values found through interpolation
    """
    sum = 0
    for point in points.values():
        sum += pow(point.get_pollution_value() - point.get_actual_pollution_value(), 2)
    rmse = math.sqrt(sum / len(points))
    return rmse


def plot_numbers(rmse_values, picked_points):
    """
    Plots Numbers on a graph
    :param rmse_values: Y- value for the graph in list form
    :param picked_points: X- values for the graph in list form
    :return:
    """
    plt.plot(picked_points, rmse_values, "ro")
    plt.show()


def run_interpolations_with_random_betas():
    """
    Runs Interpolation with random picking of of the value of Beta
    """
    rmse_values = []
    picked_points_values = []
    number_of_times = 100
    random = np.random.default_rng()

    for i in range(0, number_of_times):
        test_points = create_points_with_random_pollution(100, 100, 10)
        test_picked_points = pick_uniform_random_points(test_points, random.integers(1, 100))
        test_interpolated_points = interpolate_unknown_points(test_picked_points, test_points)
        test_rmse = root_mean_square_error(test_interpolated_points)
        rmse_values.append(test_rmse)
        picked_points_values.append(len(test_picked_points))
    plot_numbers(rmse_values, picked_points_values)


def run_interpolation_with_various_betas(points):
    """
    Runs Interpolation with number of picked points(beta) from 1 - all points picked and using uniform distribution in the picking
    """

    rmse_data = []

    for i in range(1, len(
            points) + 1):  # runs through all number of picked points starting at 1 and ending with all points picked
        sum_rmse = 0
        for j in range(0, 5):  # runs every interpolation with a certain beta 5 times and averages the results
            picked_points = pick_uniform_random_points(points, i)
            interpolated_points = interpolate_unknown_points(picked_points, points)
            sum_rmse = sum_rmse + root_mean_square_error(interpolated_points)
        rmse_data.append(sum_rmse / 5)

    plot_numbers(rmse_data, range(1, len(points) + 1))

    return rmse_data


run_interpolation_with_various_betas(create_points_with_random_pollution(100, 100, 10))
# run_interpolations_with_random_betas() #Plots points on graph

# print(Point(1, 1, 1))
#
#
# random_points1 = create_points_with_random_pollution(10, 100, 10)
# p = pick_uniform_random_points(random_points1,5)
#
# a = interpolate_unknown_points(p,random_points1)
