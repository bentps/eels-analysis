# imports
import gettext
import numpy

# local libraries
from nion.swift.model import Symbolic
from nion.eels_analysis import eels_analysis


_ = gettext.gettext


class EELSBackgroundSubtraction:
    def __init__(self, computation, **kwargs):
        self.computation = computation

    def execute(self, eels_spectrum_data_item, fit_interval_graphics, signal_interval_graphic):
        eels_spectrum_xdata = eels_spectrum_data_item.xdata
        fit_intervals = [fit_interval_graphic.interval for fit_interval_graphic in fit_interval_graphics]
        signal_interval = signal_interval_graphic.interval
        signal_xdata = eels_analysis.extract_original_signal(eels_spectrum_xdata, fit_intervals, signal_interval)
        self.__background_xdata = eels_analysis.calculate_background_signal(eels_spectrum_xdata, fit_intervals, signal_interval)
        subtracted_xdata = signal_xdata - self.__background_xdata
        offset = int(round((signal_interval[0] - fit_intervals[0][0]) * eels_spectrum_xdata.data_shape[0]))
        length = int(round((signal_interval[1] - signal_interval[0]) * eels_spectrum_xdata.data_shape[0]))
        self.__subtracted_xdata = subtracted_xdata[offset:offset + length]

    def commit(self):
        self.computation.set_referenced_xdata("background", self.__background_xdata)
        self.computation.set_referenced_xdata("subtracted", self.__subtracted_xdata)


class EELSMapping:
    def __init__(self, computation, **kwargs):
        self.computation = computation

    def execute(self, spectrum_image_data_item, fit_interval_graphics, signal_interval_graphic):
        spectrum_image_xdata = spectrum_image_data_item.xdata
        fit_intervals = [fit_interval_graphic.interval for fit_interval_graphic in fit_interval_graphics]
        signal_interval = signal_interval_graphic.interval
        self.__mapped_xdata = eels_analysis.map_background_subtracted_signal(spectrum_image_xdata, None, fit_intervals, signal_interval)

    def commit(self):
        self.computation.set_referenced_xdata("map", self.__mapped_xdata)


async def use_interval_as_signal(api, window):
    target_data_item = window.target_data_item
    target_display_item = window.target_display
    target_graphic = target_display_item.selected_graphics[0] if target_display_item and len(target_display_item.selected_graphics) == 1 else None
    target_interval = target_graphic if target_graphic and target_graphic.graphic_type == "interval-graphic" else None
    if target_data_item and target_interval:
        interval = target_interval.interval
        fit_ahead = target_display_item.add_graphic(
            {
                "type": "interval-graphic",
                "start": interval[0] * 0.8,
                "end": interval[0] * 0.9,
                "graphic_id": "background",
                "label": _("Background"),
            }
        )
        fit_behind = target_display_item.add_graphic(
            {
                "type": "interval-graphic",
                "start": interval[1] * 1.1,
                "end": interval[1] * 1.2,
                "graphic_id": "background",
                "label": _("Background"),
            }
        )
        background = api.library.create_data_item(title="{} Background".format(target_data_item.title))
        signal = api.library.create_data_item(title="{} Subtracted".format(target_data_item.title))
        computation = api.library.create_computation("eels.background_subtraction2", inputs={"eels_spectrum_data_item": target_data_item, "fit_interval_graphics": [fit_ahead, fit_behind], "signal_interval_graphic": target_interval}, outputs={"background": background, "subtracted": signal})
        computation._computation.source = target_interval._graphic
        target_interval._graphic.source = computation._computation
        fit_ahead._graphic.source = target_interval._graphic
        fit_behind._graphic.source = target_interval._graphic
        target_interval.graphic_id = "signal"
        target_interval.label = _("Signal")
        target_interval._graphic.color = "#0F0"
        target_display_item._display_item.append_display_data_channel_for_data_item(background._data_item)
        target_display_item._display_item.append_display_data_channel_for_data_item(signal._data_item)
        target_display_item._display_item.display_layers = [
            {"label": "Signal", "data_index": 2, "fill_color": "#0F0"},
            {"label": "Background", "data_index": 1, "fill_color": "rgba(255, 0, 0, 0.3)"},
            {"label": "Data", "data_index": 0, "fill_color": "#1E90FF"},
        ]
        target_display_item._display_item.set_display_property("legend_position", "top-right")


def use_signal_for_map(api, window):
    target_display = window.target_display
    target_graphic = target_display.selected_graphics[0] if target_display and len(target_display.selected_graphics) == 1 else None
    target_interval = target_graphic if target_graphic and target_graphic.graphic_type == "interval-graphic" else None
    target_data_item_ = target_display._display_item.data_items[0] if target_display and len(target_display._display_item.data_items) > 0 else None
    if target_data_item_ and target_display and target_interval:
        for computation in api.library._document_model.computations:
            if computation.processing_id == "eels.background_subtraction2" and target_interval._graphic in computation._inputs:
                fit_interval_graphics = computation.get_input("fit_interval_graphics")
                signal_interval_graphic = computation.get_input("signal_interval_graphic")
                spectrum_image = api._new_api_object(api.library._document_model.get_source_data_items(target_data_item_)[0])
                map = api.library.create_data_item_from_data(numpy.zeros_like(spectrum_image.display_xdata.data), title="{} Map".format(spectrum_image.title))
                fit_interval_graphics = [api._new_api_object(g) for g in fit_interval_graphics]
                signal_interval_graphic = api._new_api_object(signal_interval_graphic)
                computation = api.library.create_computation("eels.mapping2", inputs={"spectrum_image_data_item": spectrum_image, "fit_interval_graphics": fit_interval_graphics, "signal_interval_graphic": signal_interval_graphic}, outputs={"map": map})
                computation._computation.source = target_interval._graphic
                window.display_data_item(map)


def subtract_background_from_signal(api, window):
    window._document_controller.event_loop.create_task(use_interval_as_signal(api, window))


Symbolic.register_computation_type("eels.background_subtraction2", EELSBackgroundSubtraction)
Symbolic.register_computation_type("eels.mapping2", EELSMapping)
