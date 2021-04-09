# imports
import gettext
import numpy
import typing

# local libraries
from nion.data import Core
from nion.data import DataAndMetadata
from nion.data import Calibration
from nion.swift.model import DataStructure
from nion.swift.model import Symbolic
from nion.swift.model import Schema
from nion.swift import Facade
from nion.utils import Registry

from nion.eels_analysis import ZLP_Analysis


_ = gettext.gettext



class CalculateThickness:
    label = _("Calculate Thickness")
    inputs = {
        "eels_spectrum_data_item": {"label": _("EELS Spectrum")},
        "zlp_model": {"label": _("Zero Loss Peak Model"), "entity_id": "zlp_model"},
        }
    outputs = {
        "zlp_graphic": {"label": _("ZLP Graphic")},
        "thickness_map": {"label": _("Thickness Map")}
    }

    def __init__(self, computation, **kwargs):
        self.computation = computation
        self.__thickness = None
        self.__zlp_position = None
        self.__eels_spectrum_data_item = None

    def execute(self, eels_spectrum_data_item, zlp_model, **kwargs) -> None:
        try:
            spectrum_xdata = eels_spectrum_data_item.xdata
            assert spectrum_xdata.is_datum_1d
            assert spectrum_xdata.datum_dimensional_calibrations[0].units == "eV"
            model_xdata = None
            if zlp_model._data_structure.entity:
                entity_id = zlp_model._data_structure.entity.entity_type.entity_id
                for component in Registry.get_components_by_type("zlp-model"):
                    # print(f"{entity_id=} {component.zero_loss_peak_model_id=}")
                    if entity_id == component.zero_loss_peak_model_id:
                        fit_result = component.fit_zero_loss_peak(spectrum_xdata=spectrum_xdata)
                        model_xdata = fit_result["zero_loss_peak_model"]
            if model_xdata is not None:
                self.__thickness = numpy.log(numpy.sum(spectrum_xdata.data, axis=-1) / numpy.sum(model_xdata.data, axis=-1))
                if numpy.ndim(self.__thickness) > 0:
                    self.__zlp_position = None
                    self.__thickness = DataAndMetadata.new_data_and_metadata(self.__thickness,
                                                                             dimensional_calibrations=spectrum_xdata.dimensional_calibrations[:-1],
                                                                             intensity_calibration = Calibration.Calibration(units="1/\N{GREEK SMALL LETTER LAMDA}"))
                else:
                    self.__zlp_position = numpy.argmax(model_xdata.data)
            else:
                self.__thickness = None
                self.__zlp_position = None
            self.__eels_spectrum_data_item = eels_spectrum_data_item
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print(e)
            raise

    def commit(self):
        if self.__thickness is not None:
            if self.__zlp_position is None:
                self.computation.set_referenced_xdata("thickness_map", self.__thickness)
            else:
                zlp_graphic = self.computation.get_result("zlp_graphic", None)
                data_length = self.__eels_spectrum_data_item.data.shape[-1]
                if not zlp_graphic:
                    zlp_graphic = self.__eels_spectrum_data_item.add_channel_region(self.__zlp_position / data_length)
                    self.computation.set_result("zlp_graphic", zlp_graphic)
                zlp_graphic.position = self.__zlp_position / data_length
                zlp_graphic.graphic_id = "zlp_graphic"
                zlp_graphic._graphic.color = "#0F0"
                zlp_graphic.label = f"Sample thickness: {self.__thickness:.4g} 1/\N{GREEK SMALL LETTER LAMDA}"
        else:
            zlp_graphic = self.computation.get_result("zlp_graphic", None)
            if zlp_graphic:
                self.__eels_spectrum_data_item.remove_region(zlp_graphic)
                self.computation.set_result("zlp_graphic", None)


def measure_thickness(api: Facade.API_1, window: Facade.DocumentWindow) -> None:
    target_data_item = window.target_data_item
    if not target_data_item:
        return

    zlp_model = DataStructure.DataStructure(structure_type="simple_peak_model")
    api.library._document_model.append_data_structure(zlp_model)
    zlp_model.source = target_data_item._data_item
    thickness_map_data_item = None
    if target_data_item.xdata and target_data_item.xdata.is_navigable:
        thickness_map_data_item = api.library.create_data_item(title=f"{target_data_item.title} thickness map")

    api.library.create_computation("eels.calculate_thickness",
                                   inputs={
                                       "eels_spectrum_data_item": target_data_item,
                                       "zlp_model": api._new_api_object(zlp_model),
                                   },
                                   outputs={
                                       "zlp_graphic": None,
                                       "thickness_map": thickness_map_data_item}
                                   )

Symbolic.register_computation_type("eels.calculate_thickness", CalculateThickness)
