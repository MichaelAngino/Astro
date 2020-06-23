import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct as DP
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


# test comment

# np.random.multivariate_normal([0, 0], [[1,1], [1,1]] , 1)[0]
# with vars names: np.random.multivariate_normal(mean_vector_of_each_point=[0, 0], COV=[[1,1], [1,1]] , nb_maps=1)[id_map=0]

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
        self.pollution_value = new_pollution_value

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


def distance(first_point, second_point):
    return math.sqrt(math.pow(first_point.x - second_point.x, 2) + math.pow(first_point.y - second_point.y, 2))


def create_points_with_random_pollution_1d(length, mean, std):
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


def create_points_with_spatially_correlated_pollution_2d(side_length, mean, length_scale, num_maps):
    '''
    Creates a map of pollution values determined by their spatial correlation
    :param side_length: Number of points on each side of the matrix
    :param mean: Mean of the gaussian distribution of pollution values
    :param length_scale: The length scale of the RBF kernel
    :param num_maps: Number of pollution maps used in data
    '''
    # with vars names: np.random.multivariate_normal(mean_vector_of_each_point=[0, 0], COV=[[1,1], [1,1]] , nb_maps=1)[id_map=0]
    mean_vector = []
    for i in range(side_length * side_length):
        mean_vector.append(mean)

    initial_point_map = {}
    x = 5
    label_index = 0
    for i in range(0, side_length):
        y = 5
        for j in range(0, side_length):
            initial_point_map[label_index] = Point(label_index, 'NaN', x, y)
            label_index += 1
            y += 10
        x += 10

    new_map = {}
    covariance_matrix = create_covariance_matrix(initial_point_map, length_scale)
    '''DEBUGGING
    print(len(initial_point_map))
    for i in range(len(covariance_matrix)):
        for j in range(len(covariance_matrix[0])):
            print(covariance_matrix[i][j], end=' ')
        print()
    # print(str(len(mean_vector)) + " "+ str(len(covariance_matrix)))
    current = (np.random.multivariate_normal(mean_vector, covariance_matrix, num_maps))
    for i in range(len(current)):
        print(current[i])
    print("END OF DEBUG")
    '''
    new_map = np.random.multivariate_normal(mean_vector, covariance_matrix, num_maps)
    return new_map



def create_covariance_matrix(points,  length_scale):
    """
    Creates Covariance Matrix
    :param points: Map of all points
    :param length_scale: Length Scale
    :return: A matrix
    """

    covariance = []
    for i in range(0, len(points)):
        covariance.append([])
        for j in range(0, len(points)):
            covariance[i].append(
                np.exp(-np.power(distance(points[i], points[j]), 2) / (2 * length_scale * length_scale)))

    return covariance


def interpolate_points_using_positions(known_points, wanted_point_positions, kernel=RBF(10, (1e-2, 1e2)) * C(1),
                                       fixed=False):
    """
     Predicts points based on known data using Kriging (Gaussian Processes)
    :param known_points: list of points
    :param wanted_point_positions: list of wanted point posistions [[x1,y1], [x2,y2]]
    :param kernel:  Kernal to use in interpolation
    :param fixed:  True = no opitimization of hyperparamater, False = optimization of hyperparamter
    :return: a list of all predicted pollution values
    """

    # kernel = DP(1)
    # kernel = RBF(10, (1e-2, 1e2)) * C(1)


    if fixed:
        gp = GaussianProcessRegressor(kernel, alpha=10, n_restarts_optimizer=4, optimizer= None)  # Instantiate a fixed Gaussian Process model
    else:
        gp = GaussianProcessRegressor(kernel, alpha=10, n_restarts_optimizer=4)  # Instantiate a Gaussian Process model

    known_points_position_list = to_list_of_positions(known_points)

    pollution_value_list = []  # converts point list into a list of pollution values
    for point in known_points.values():
        pollution_value_list.append(point.get_pollution_value())

    gp.fit(known_points_position_list, pollution_value_list)  # Fits model to data

    #
    # prediction_list = known_points_position_list
    # prediction_list.extend(wanted_point_positions)
    # return gp.predict(prediction_list)[len(known_points):]  # predicts on new data


    return (gp.predict(wanted_point_positions), gp.get_params())


def interpolate_unknown_points(known_points, all_points, kernel = RBF(10, (1e-2, 1e2)) * C(1), fixed = False):
    """
    Interpolate pollution values for points that are have not been measured
    :param known_points: A Dictionary of all points that have been measured {label:point}
    :param all_points: A Dictionary of all points that exist {Label: point}
    :param kernel:  Kernal to use in interpolation
    :param fixed:  True = no opitimization of hyperparamater, False = optimization of hyperparamter
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
                                                                       kernel, fixed)  # interpolates the pollution values for the posistions where we have not measured pollution values


    interpolated_map = copy_dictionary_with_points(known_points)  # creates a copy of the known_points dictionary

    for i in range(0, len(
            unknown_labels)):  # adds missing points and their interpolated pollution values into the new dictionary
        interpolated_map[unknown_labels[i]] = all_points.get(unknown_labels[i]).copy()
        interpolated_map[unknown_labels[i]].set_pollution_value(interpolated_pollution_values[i])

    return (interpolated_map, length_scale)

def interpolate_unknown_points_of_a_map_of_maps_of_points(known_points, all_points, kernel =RBF(10, (1e-2, 1e2)) * C(1), fixed = False):
    """

    :param known_points: A map of maps of all points that have been measured
    :param all_points:  A  map of maps of all points that exist
    :param kernel:  Kernal to use in interpolation
    :param fixed:  True = no opitimization of hyperparamater, False = optimization of hyperparamter
    :return: A tuple of (A new map of maps of interpolated values, length of kernel)
    """

    interpolated_maps = {}
    for label in all_points.keys():
        interpolated_maps[label] = interpolate_unknown_points(known_points, all_points, kernel, fixed)

    return interpolated_maps






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
        if not random_num in random_picks and 0 < random_num < len(
                points):  # Checks to make sure number hasnt been picked yet and is within range
            random_picks.append(random_num)
    new_map = {}
    for i in random_picks:  # assigns and copies picked points into a new dictionary
        new_map[i] = Point(i, points.get(i).get_actual_pollution_value(), points.get(i).get_x_cord())
        new_map.get(i).read_pollution_value()

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
    plt.xlabel("Number of Known Measurements")
    plt.ylabel("RMSE")
    plt.show()


def run_interpolations_with_random_betas_1d():
    """
    Runs Interpolation with random picking of the value of Beta
    """
    rmse_values = []
    picked_points_values = []
    number_of_times = 100  # this will be the number of points displayed on the graph
    random = np.random.default_rng()

    for i in range(0, number_of_times):
        test_points = create_points_with_random_pollution_1d(100, 100, 10)
        test_picked_points = pick_uniform_random_points(test_points, random.integers(1,
                                                                                     100))  # picks a random number of known points
        test_interpolated_points = interpolate_unknown_points(test_picked_points, test_points)
        test_rmse = root_mean_square_error(test_interpolated_points)
        rmse_values.append(test_rmse)
        picked_points_values.append(len(test_picked_points))

    plot_numbers(rmse_values, picked_points_values)


def run_interpolation_with_various_betas(points, kernel=RBF(10, (1e-2, 1e2)) * C(1)):
    """
    Runs Interpolation with number of picked points(beta) from 1 to all points-1 picked and using uniform distribution in
    the picking in 1 Dimesnsion
    """

    rmse_data = []

    for i in range(1, len(
            points)):  # runs through all number of picked points starting at 1 and ending with all points picked-1
        sum_rmse = 0
        for j in range(0, 3):  # runs every interpolation with a certain beta 5 times and averages the results
            picked_points = pick_uniform_random_points(points, i)
            interpolated_points = interpolate_unknown_points(picked_points, points, kernel)
            sum_rmse = sum_rmse + root_mean_square_error(interpolated_points)
        rmse_data.append(sum_rmse / 5)

    plot_numbers(rmse_data, range(1, len(points)))

    return rmse_data


def see_what_its_doing_1d():
    """
    Graphs all points and interpolates unknown points, useful for visualizing Gaussian Interpolation and affects of kernals
    :return:
    """
    all_points = create_points_with_random_pollution_1d(100, 100, 10)
    picked_points = pick_uniform_random_points(all_points, 20)
    interpolated_points = interpolate_unknown_points(picked_points, all_points)

    picked_x = []
    picked_pollution = []
    for label, point in picked_points.items():
        picked_x.append(label)
        picked_pollution.append(point.get_pollution_value())

    interp_x = []
    inter_pollution = []

    for label, point in interpolated_points.items():
        if not label in picked_x:
            interp_x.append(label)
            inter_pollution.append(point.get_pollution_value())

    plt.plot(picked_x, picked_pollution, "ro", interp_x, inter_pollution, "go")
    plt.xlabel("Point Label")
    plt.ylabel("Pollution Value")
    plt.show()


# see_what_its_doing_1d()
# run_interpolation_with_various_betas(create_points_with_random_pollution_1d(100, 100, 10))

# random_total_points_2d = create_points_with_random_pollution_2d(10, 100, 10)

# run_interpolation_with_various_betas(random_total_points_2d, RBF(10, (1e-2, 1e2)) * C(1))
# print(to_list_of_positions(random_total_points_2d))
# run_interpolation_with_various_betas(random_total_points_2d, DP(1))
# run_interpolations_with_random_betas() #Plots points on graph

# see_what_its_doing()

# print(Point(1, 1, 1))
#
#
# random_points1 = create_points_with_random_pollution(10, 100, 10)
# p = pick_uniform_random_points(random_points1,5)
#
# a = interpolate_unknown_points(p,random_points1)
test_points = create_points_with_spatially_correlated_pollution_2d(10, 100, 1, 1)
for i in range(len(test_points)):
    print(test_points[i], end=' ')
