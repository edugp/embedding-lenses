from typing import Dict

import numpy as np
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Cividis256, Palette
from bokeh.plotting import Figure, figure
from bokeh.transform import factor_cmap


def draw_interactive_scatter_plot(
    hover_fields: Dict[str, np.ndarray], xs: np.ndarray, ys: np.ndarray, values: np.ndarray, palette: Palette = Cividis256
) -> Figure:
    # Assign a color for each value based on its 256-quantile (since color palette has 256 values)
    quantiles = np.quantile(values, np.arange(0, 1, 1 / 255))
    values_color = np.searchsorted(quantiles, values, side="right")
    values_color_set = np.sort(values_color).astype(int).astype(str)

    values_list = values.astype(str).tolist()
    values_set = np.sort(values).astype(str).tolist()

    data = {"_x_": xs, "_y_": ys, "_label_color_": values_list}
    hover_data = {field: values.astype(str).tolist() for field, values in hover_fields.items()}
    data.update(hover_data)
    source = ColumnDataSource(data=data)
    hover = HoverTool(tooltips=[(field, "@%s{safe}" % field) for field in hover_data.keys()])
    p = figure(plot_width=800, plot_height=800, tools=[hover])
    p.circle(
        "_x_",
        "_y_",
        size=10,
        source=source,
        fill_color=factor_cmap("_label_color_", palette=[palette[int(id_)] for id_ in values_color_set], factors=values_set),
    )

    p.axis.visible = False
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.toolbar.logo = None
    return p
