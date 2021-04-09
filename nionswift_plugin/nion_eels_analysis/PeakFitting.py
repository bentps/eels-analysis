from __future__ import annotations

# imports
import gettext
import numpy
import typing

# local libraries
from nion.data import Core
from nion.data import DataAndMetadata
from nion.swift.model import DataStructure
from nion.swift.model import Symbolic
from nion.swift.model import Schema
from nion.swift import Facade
from nion.utils import Registry

from nion.eels_analysis import ZLP_Analysis


_ = gettext.gettext


class FitZeroLossPeak:
    label = _("Zero Loss Peak Subtraction")
    inputs = {
        "eels_spectrum_data_item": {"label": _("EELS Spectrum")},
        "zlp_model": {"label": _("Zero Loss Peak Model"), "entity_id": "zlp_model"},
        "show_zlp_stats": {"label": _("Show ZLP stats")}
        }
    outputs = {
        "zero_loss_peak": {"label": _("Background")},
        "subtracted": {"label": _("Subtracted")},
        "zlp_interval": {"label": _("ZLP Interval")}
    }

    def __init__(self, computation, **kwargs):
        self.computation = computation
        self.__model_xdata = None
        self.__subtracted_xdata = None
        self.__zlp_interval_bounds = None
        self.__eels_spectrum_data_item = None

    def execute(self, eels_spectrum_data_item, zlp_model, show_zlp_stats, **kwargs) -> None:
        try:
            spectrum_xdata = eels_spectrum_data_item.xdata
            assert spectrum_xdata.is_datum_1d
            assert spectrum_xdata.datum_dimensional_calibrations[0].units == "eV"
            eels_spectrum_xdata = spectrum_xdata
            model_xdata = None
            subtracted_xdata = None
            zlp_position = None
            zlp_fwhm = None
            if zlp_model._data_structure.entity:
                entity_id = zlp_model._data_structure.entity.entity_type.entity_id
                for component in Registry.get_components_by_type("zlp-model"):
                    # print(f"{entity_id=} {component.zero_loss_peak_model_id=}")
                    if entity_id == component.zero_loss_peak_model_id:
                        fit_result = component.fit_zero_loss_peak(spectrum_xdata=spectrum_xdata)
                        model_xdata = fit_result["zero_loss_peak_model"]
                        # use 'or' to avoid doing subtraction if subtracted_spectrum already present
                        subtracted_xdata = fit_result.get("subtracted_spectrum", None) or Core.calibrated_subtract_spectrum(spectrum_xdata, model_xdata)
                        zlp_stats = fit_result.get("zlp_stats", dict())
                        zlp_position = zlp_stats.get("position")
                        zlp_fwhm = zlp_stats.get("fwhm")
            if model_xdata is None:
                model_xdata = DataAndMetadata.new_data_and_metadata(numpy.zeros_like(eels_spectrum_xdata.data), intensity_calibration=eels_spectrum_xdata.intensity_calibration, dimensional_calibrations=eels_spectrum_xdata.dimensional_calibrations)
            if subtracted_xdata is None:
                subtracted_xdata = DataAndMetadata.new_data_and_metadata(eels_spectrum_xdata.data, intensity_calibration=eels_spectrum_xdata.intensity_calibration, dimensional_calibrations=eels_spectrum_xdata.dimensional_calibrations)

            if show_zlp_stats:
                if zlp_position is None or zlp_fwhm is None:
                    _, _, left, right = ZLP_Analysis.estimate_zlp_amplitude_position_width_com(model_xdata.data)
                else:
                    left, right = (zlp_position - 0.5 * zlp_fwhm, zlp_position + 0.5 * zlp_fwhm)
                self.__zlp_interval_bounds = (left, right)
            else:
                self.__zlp_interval_bounds = None

            self.__model_xdata = model_xdata
            self.__subtracted_xdata = subtracted_xdata
            self.__eels_spectrum_data_item = eels_spectrum_data_item
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print(e)
            raise

    def commit(self):
        self.computation.set_referenced_xdata("zero_loss_peak", self.__model_xdata)
        self.computation.set_referenced_xdata("subtracted", self.__subtracted_xdata)
        if self.__zlp_interval_bounds is not None:
            start, end = self.__zlp_interval_bounds
            data_length = self.__eels_spectrum_data_item.data.shape[-1]
            zlp_interval = self.computation.get_result("zlp_interval", None)
            if not zlp_interval:
                zlp_interval = self.__eels_spectrum_data_item.add_interval_region(start / data_length, end / data_length)
                self.computation.set_result("zlp_interval", zlp_interval)
            zlp_interval.interval = start / data_length, end / data_length
            zlp_interval.graphic_id = "zlp_interval"
            zlp_interval._graphic.color = "#0F0"
            cal = self.__eels_spectrum_data_item.xdata.dimensional_calibrations[-1]
            zlp_interval.label = (f"Position: {cal.convert_to_calibrated_value_str(start + 0.5 * (end - start))}, \n"
                                  f"FWHM: {cal.convert_to_calibrated_size_str(end - start)}")
        else:
            zlp_interval = self.computation.get_result("zlp_interval", None)
            if zlp_interval:
                self.__eels_spectrum_data_item.remove_region(zlp_interval)
                self.computation.set_result("zlp_interval", None)


def add_peak_fitting_computation(api: Facade.API_1, library: Facade.Library, display_item: Facade.Display, data_item: Facade.DataItem) -> None:
    zero_loss_peak = api.library.create_data_item(title="{} Zero Loss Peak".format(data_item.title))
    signal = api.library.create_data_item(title="{} Subtracted".format(data_item.title))

    zlp_model = DataStructure.DataStructure(structure_type="simple_peak_model")
    library._document_model.append_data_structure(zlp_model)
    zlp_model.source = zero_loss_peak._data_item

    api.library.create_computation("eels.fit_zlp",
                                   inputs={
                                       "eels_spectrum_data_item": data_item,
                                       "zlp_model": api._new_api_object(zlp_model),
                                       "show_zlp_stats": True,
                                   },
                                   outputs={
                                       "zero_loss_peak": zero_loss_peak,
                                       "subtracted": signal,
                                       "zlp_interval": None}
                                   )
    display_item._display_item.append_display_data_channel_for_data_item(zero_loss_peak._data_item)
    display_item._display_item.append_display_data_channel_for_data_item(signal._data_item)
    display_item._display_item.move_display_layer_at_index_backward(0)
    display_item._display_item.move_display_layer_at_index_backward(1)
    display_item._display_item._set_display_layer_properties(0, label=_("Zero Loss Peak"),
                                                             fill_color="rgba(255, 0, 0, 0.2)",
                                                             stroke_color="rgba(255, 0, 0, 1.0)")
    display_item._display_item._set_display_layer_properties(1, label=_("Subtracted"),
                                                             fill_color="rgba(0, 255, 0, 0.2)",
                                                             stroke_color="rgba(0, 255, 0, 1.0)")
    display_item._display_item._set_display_layer_properties(2, label=_("Raw Data"), fill_color="#1E90FF")
    display_item._display_item.set_display_property("legend_position", "top-right")


def fit_zero_loss_peak(api: Facade.API_1, window: Facade.DocumentWindow) -> None:
    target_data_item = window.target_data_item
    target_display_item = window.target_display
    if target_data_item:
        add_peak_fitting_computation(api, window.library, target_display_item, target_data_item)


Symbolic.register_computation_type("eels.fit_zlp", FitZeroLossPeak)

ZeroLossPeakModel = Schema.entity("zlp_model", None, None, {})


def component_registered(component, component_types):
    if "zlp-model" in component_types:
        # when a background model is registered, create an empty (for now) entity type, and register it with the data
        # structure so that an entity for use with the UI and computations can be created when the data structure loads.
        zlp_model_entity = Schema.entity(component.zero_loss_peak_model_id, ZeroLossPeakModel, None, {})
        DataStructure.DataStructure.register_entity(zlp_model_entity, entity_name=component.title, entity_package_name=component.package_title)


_component_registered_listener = Registry.listen_component_registered_event(component_registered)

# handle any components that have already been registered.
for component in Registry.get_components_by_type("zlp-model"):
    component_registered(component, {"zlp-model"})
