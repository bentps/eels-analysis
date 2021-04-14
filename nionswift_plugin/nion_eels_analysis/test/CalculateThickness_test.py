import numpy
import unittest

from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.swift import Application
from nion.swift import Facade
from nion.swift.model import DataItem
import time

from nion.swift.test import TestContext
from nion.ui import TestUI

from nionswift_plugin.nion_eels_analysis import CalculateThickness


Facade.initialize()


class TestCalculateThickness(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def __run_until_complete(self, document_controller):
        # run for 0.2s; recomputing and running periodic
        for _ in range(10):
            document_controller.document_model.recompute_all()
            document_controller.periodic()
            time.sleep(1/50)

    def test_calculate_thickness_for_single_spectrum(self):
        with TestContext.create_memory_context() as profile_context:
            document_controller = profile_context.create_document_controller_with_application()
            document_model = document_controller.document_model
            data = numpy.ones((100,), dtype=numpy.float32)
            data[10] = 100
            intensity_calibration = Calibration.Calibration(units="~")
            dimensional_calibrations = [Calibration.Calibration(offset= -10.0, scale=1.0, units="eV")]
            data_descriptor = DataAndMetadata.DataDescriptor(is_sequence=False, collection_dimension_count=0, datum_dimension_count=1)
            xdata = DataAndMetadata.new_data_and_metadata(data, intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations, data_descriptor=data_descriptor)
            data_item = DataItem.new_data_item(xdata)
            document_model.append_data_item(data_item)
            display_item = document_model.get_display_item_for_data_item(data_item)
            document_controller.select_display_items_in_data_panel([display_item])
            document_controller.data_panel_focused()
            api = Facade.get_api("~1.0", "~1.0")
            CalculateThickness.measure_thickness(api, api.application.document_windows[0])
            self.__run_until_complete(document_controller)
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(1, len(api.library.data_items))
            # measure_thickness should create one graphic
            self.assertEqual(1, len(api.library.data_items[0].graphics))
            # It should find the peak at 10 and put the graphic there
            self.assertAlmostEqual(api.library.data_items[0].graphics[0].position, 0.1)
            # Check that the correct thickness is shown in the label
            label = api.library.data_items[0].graphics[0].label
            thickness = float(label.split()[2])
            self.assertAlmostEqual(numpy.log(numpy.sum(data)/numpy.amax(data)), thickness, places=1)
            # Test cleanup
            document_model.remove_data_item(data_item)
            self.assertEqual(0, len(document_model.data_items))
            self.assertEqual(0, len(document_model.display_items))
            self.assertEqual(0, len(document_model.data_structures))

    def test_calculate_thickness_for_spectrum_image(self):
        with TestContext.create_memory_context() as profile_context:
            document_controller = profile_context.create_document_controller_with_application()
            document_model = document_controller.document_model
            data = numpy.ones((4, 3, 100), dtype=numpy.float32)
            data[..., 10] = 100
            intensity_calibration = Calibration.Calibration(units="~")
            dimensional_calibrations = [Calibration.Calibration(), Calibration.Calibration(), Calibration.Calibration(offset= -10.0, scale=1.0, units="eV")]
            data_descriptor = DataAndMetadata.DataDescriptor(is_sequence=False, collection_dimension_count=2, datum_dimension_count=1)
            xdata = DataAndMetadata.new_data_and_metadata(data, intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations, data_descriptor=data_descriptor)
            data_item = DataItem.new_data_item(xdata)
            document_model.append_data_item(data_item)
            display_item = document_model.get_display_item_for_data_item(data_item)
            document_controller.select_display_items_in_data_panel([display_item])
            document_controller.data_panel_focused()
            api = Facade.get_api("~1.0", "~1.0")
            CalculateThickness.measure_thickness(api, api.application.document_windows[0])
            self.__run_until_complete(document_controller)
            # measure_thickness should create a data item (the thickness map)
            self.assertEqual(2, len(document_model.data_items))
            self.assertEqual(2, len(document_model.display_items))
            self.assertEqual(2, len(api.library.data_items))
            # measure_thickness should not create any graphics
            self.assertEqual(0, len(api.library.data_items[0].graphics))
            result_data_item = api.library.data_items[1]
            # Check that the thickness map has the correct shape
            self.assertSequenceEqual(data.shape[:-1], result_data_item.data.shape)
            # Check that the correct thickness is shown in the thickness map
            self.assertTrue(numpy.allclose(result_data_item.data, numpy.log(numpy.sum(data, axis=-1)/numpy.amax(data, axis=-1)), atol=0.1))
            # Test cleanup
            document_model.remove_data_item(data_item)
            self.assertEqual(0, len(document_model.data_items))
            self.assertEqual(0, len(document_model.display_items))
            self.assertEqual(0, len(document_model.data_structures))