import unittest
from collections import defaultdict
import Pollution


class MyTestCase(unittest.TestCase):
    def test_point_list(self):
        """
        Testing Create_list_of_points method
        :return:
        """
        test_list = Pollution.create_list_of_random_pollution_points(100,100,0)
        self.assertEqual(100, len(test_list))
        for i in range(0,len(test_list)):
            self.assertEqual(100,test_list[i].get_pollution_value())

        self.assertEqual(5, test_list[0].get_x_cord())
        self.assertEqual(995, test_list[99].get_x_cord())

    def test_point(self):
        """
        Testing Point Class
        :return:
        """
        test_point = Pollution.Point(1,100,0)
        self.assertEqual(1,test_point.get_label())
        self.assertEqual(100,test_point.get_pollution_value())
        test_point.set_pollution_value(20)

        self.assertEqual(20, test_point.get_pollution_value())

    def test_uniform_point_selection(self):
        """
        Tests uniform point selction
        :return:
        """

        test_points = Pollution.create_points_with_random_pollution(10,10,1)
        counter = defaultdict(int)
        num_of_runs = 10000

        for i in range(0,num_of_runs):
            a = Pollution.pick_uniform_random_points(test_points, 5)
            for key in a.keys():
                counter[key] = counter[key]+1

        for key in counter.keys():
            counter[key] = counter[key]/ num_of_runs

        for i in range(0,9):
            self.assertAlmostEqual(.5, counter[i],None,None,.02)







