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

    def run_experiment_with_various_length_scales_linear(bottom_bound, top_bound, side_length, mean, standard_deviation,
                                                         pick_number, number_of_maps, step):
        """
        Experiment to see rmse of cheating and not cheating regreassion on varous length scales (traverses linearly)  in 2D. Uses uniform point selection and RBF kernel
        :param step: step
        :param bottom_bound: bottom bound of length scale
        :param top_bound: top bound of length scale not inclusive
        :param side_length: number of points on one side of the square of points
        :param mean: Mean pollution value to be set
        :param standard_deviation: Standard Deviation
        :param pick_number: the Beta (the number of points to select to be measured)
        :param number_of_maps: Number of trials for each length scale
        :return:
        """
        not_cheating_data = []
        cheating_data = []
        for length_scale in range(bottom_bound, top_bound, step):  # runs through each length scale
            points = create_points_with_spatially_correlated_pollution_2d(side_length, mean, standard_deviation,
                                                                          length_scale,
                                                                          number_of_maps)  # Creates all points
            picked_points = pick_uniform_random_points_on_map_of_maps(points,
                                                                      pick_number)  # Picks points to be measured
            interpolated_points = interpolate_unknown_points_of_a_map_of_maps_of_points(picked_points, points,
                                                                                        # Interpolates using noncheating method
                                                                                        RBF(np.random.randint(1e-05,
                                                                                                              100 + 1)),
                                                                                        fixed=False)

            not_cheating_data.append(
                average_rmse_of_maps(
                    interpolated_points))  # adds average rms of all the trials for the noncheating method
            interpolated_points = interpolate_unknown_points_of_a_map_of_maps_of_points(picked_points, points,
                                                                                        # Interpolates using cheating method
                                                                                        RBF(length_scale,
                                                                                            ), fixed=True)

            cheating_data.append(
                average_rmse_of_maps(
                    interpolated_points))  # adds average rmse of all the trials for the cheating method
            # print(length_scale)

        plot_numbers(range(bottom_bound, top_bound, step), not_cheating_data, range(bottom_bound, top_bound, step),
                     cheating_data,  # Plots the data Red is not cheating, Green Cheating
                     "Length Scale", "RMSE")

    def run_experiment_with_various_length_scales_log(bottom_bound, top_bound, side_length, mean, std, pick_number,
                                                      number_of_maps):
        """
        Experiment to see rmse of cheating and not cheating regreassion on varous length scales (traverses by powers of 10)  in 2D. Uses uniform point selection and RBF kernel
        :param bottom_bound: bottom bound of length scale
        :param top_bound: top bound of length scale not inclusive
        :param side_length: number of points on one side of the square of points
        :param mean: Mean pollution value to be set
        :param std: Standard Deviation
        :param pick_number: the Beta (the number of points to select to be measured)
        :param number_of_maps: Number of trials for each length scale
        :return:
        """
        not_cheating_data = []
        cheating_data = []
        length_scale = bottom_bound
        length_scale_list = []
        while length_scale <= top_bound:  # runs through each length scale
            points = create_points_with_spatially_correlated_pollution_2d(side_length, mean, std, length_scale,
                                                                          number_of_maps)  # Creates all points
            picked_points = pick_uniform_random_points_on_map_of_maps(points, pick_number, 0,
                                                                      std)  # Picks points to be measured
            interpolated_points = interpolate_unknown_points_of_a_map_of_maps_of_points(picked_points, points,
                                                                                        # Interpolates using noncheating method
                                                                                        RBF(np.random.randint(1e-05,
                                                                                                              100 + 1)),
                                                                                        fixed=False)

            not_cheating_data.append(
                average_rmse_of_maps(
                    interpolated_points))  # adds average rms of all the trials for the noncheating method
            interpolated_points = interpolate_unknown_points_of_a_map_of_maps_of_points(picked_points, points,
                                                                                        # Interpolates using cheating method
                                                                                        RBF(length_scale,
                                                                                            ), fixed=True)

            cheating_data.append(
                average_rmse_of_maps(
                    interpolated_points))  # adds average rmse of all the trials for the cheating method
            length_scale_list.append(length_scale)
            length_scale = length_scale * 10

        plot_numbers(length_scale_list, not_cheating_data, length_scale_list, cheating_data,
                     # Plots the data Red is not cheating, Green Cheating
                     "Length Scale", "RMSE", x_log_scale=True)

    def run_experiment_with_varied_standard_deviations(bottom_bound, top_bound, steps, side_length, mean,
                                                       std_of_pollution,
                                                       std_deviation_values_of_measurment, pick_number, num_maps):
        """
        Experiment to see relation between RMSE and cheating regression data when varying the standard_deviation of pollution values
        :param bottom_bound: lower bound of length scale
        :param top_bound: top bound of length scale
        :param steps: difference between each adjacent length scale
        :param side_length: number of points on one side of the square of points
        :param mean: mean pollution value
        :param std_deviation_values_of_measurment: list of various standard deviations to be used and plotted
        :param pick_number: number of selected known points
        :param num_maps: number of maps used in the averaging process
        :return:
        """

        data_with_varied_std_deviations = []
        length_scale_list = list_of_length_scales(bottom_bound, top_bound, steps)

        for i in range(len(std_deviation_values_of_measurment)):
            cheating_data = []
            for j in range(len(length_scale_list)):
                points = create_points_with_spatially_correlated_pollution_2d(side_length, mean, std_of_pollution,
                                                                              length_scale_list[j], num_maps)
                picked_points = pick_uniform_random_points_on_map_of_maps(points,
                                                                          pick_number, 0,
                                                                          std_deviation_values_of_measurment[
                                                                              i])  # Picks points to be measured
                interpolated_points = interpolate_unknown_points_of_a_map_of_maps_of_points(picked_points, points,
                                                                                            RBF(length_scale_list[j],
                                                                                                ),
                                                                                            fixed=True)  # cheating method
                cheating_data.append(average_rmse_of_maps(interpolated_points))
                print("LengthScale: ", length_scale_list[j], " finished")

            print("Standard Deviation: ", i, " finished")
            data_with_varied_std_deviations.append(cheating_data)

        plot_varied_std_deviations(length_scale_list, data_with_varied_std_deviations, "Length Scale", "RMSE")


def see_what_its_doing_2d(length_scale, cheating, pollution_mean, pollution_std, pick_number):
    """
    3D graphs the pollution value of the measured and interpolated pollution values
    :param pick_number:
    :param pollution_std:
    :param pollution_mean:
    :param length_scale:
    :param cheating:
    :return:
    """

    a = create_points_with_spatially_correlated_pollution_2d(10, 100, 10, length_scale, 1)
    b = pick_uniform_random_points_on_map_of_maps(a, pick_number, pollution_mean, pollution_std)
    if cheating:
        c = interpolate_unknown_points_of_a_map_of_maps_of_points(b, a, RBF(length_scale), fixed=True)
    else:
        c = interpolate_unknown_points_of_a_map_of_maps_of_points(b, a, RBF(np.random.randint(1, 10000)), fixed=False)

    x1 = []
    y1 = []
    z1 = []
    for point in b[0].values():
        x1.append(point.get_x_cord())
        y1.append(point.get_y_cord())
        z1.append(point.get_pollution_value())

    x2 = []
    y2 = []
    z2 = []

    for label, point in c[0][0].items():
        if not label in b[0].keys():
            x2.append(point.get_x_cord())
            y2.append(point.get_y_cord())
            z2.append(point.get_pollution_value())
    print(average_rmse_of_maps(c))
    plot_numbers_3d_and_save(x1, y1, z1, x2, y2, z2, "Rotating Graph.gif")

    # mywriter = animation.FFMpegWriter(fps=60)
    # rot_animation.save("rotation.mp4",dpi = 80, writer= mywriter)


def see_what_its_doing_2d_comparison(length_scale, true_values=False):
    """
    ALlows visual comparison between interpolation with a cheating and not cheating interpolation
    :param length_scale:
    :return:
    """

    a = create_points_with_spatially_correlated_pollution_2d(10, 100, 10, length_scale, 1)
    b = pick_uniform_random_points_on_map_of_maps(a, 20)
    c1 = interpolate_unknown_points_of_a_map_of_maps_of_points(b, a, RBF(length_scale), fixed=True)
    c2 = interpolate_unknown_points_of_a_map_of_maps_of_points(b, a, RBF(np.random.randint(1, 10000)), fixed=False)

    x1 = []
    y1 = []
    z1 = []
    for point in b[0].values():
        x1.append(point.get_x_cord())
        y1.append(point.get_y_cord())
        z1.append(point.get_pollution_value())

    x2_fixed = []
    y2_fixed = []
    z2_fixed = []

    for label, point in c1[0][0].items():
        if not label in b[0].keys():
            x2_fixed.append(point.get_x_cord())
            y2_fixed.append(point.get_y_cord())
            z2_fixed.append(point.get_pollution_value())

    x2_not_fixed = []
    y2_not_fixed = []
    z2_not_fixed = []

    for label, point in c2[0][0].items():
        if not label in b[0].keys():
            x2_not_fixed.append(point.get_x_cord())
            y2_not_fixed.append(point.get_y_cord())
            z2_not_fixed.append(point.get_pollution_value())

    if true_values:
        x3_true_values = []
        y3_true_values = []
        z3_true_values = []

        for label, point in a[0].items():
            if not label in b[0].keys():
                x3_true_values.append(point.get_x_cord())
                y3_true_values.append(point.get_y_cord())
                z3_true_values.append(point.get_actual_pollution_value())

        plot_numbers_3d_and_save(x3_true_values, y3_true_values, z3_true_values, x2_fixed, y2_fixed, z2_fixed,
                                 "True Value Comparison Fixed Graph.gif")
        plot_numbers_3d_and_save(x3_true_values, y3_true_values, z3_true_values, x2_not_fixed, y2_not_fixed,
                                 z2_not_fixed, "True value Not Fixed Graph.gif")

    plot_numbers_3d_and_save(x1, y1, z1, x2_fixed, y2_fixed, z2_fixed, "Fixed Rotating Graph.gif")
    plot_numbers_3d_and_save(x1, y1, z1, x2_not_fixed, y2_not_fixed, z2_not_fixed, "Not Fixed Rotating Graph.gif")
