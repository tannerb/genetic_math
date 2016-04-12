# graph
from datetime import date

import numpy as np

from bokeh.client import push_session
from bokeh.io import output_server, show, vform
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc, vplot, output_server
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn


from random import randint

# create a plot and style its properties
p = figure(x_range=(0, 100), y_range=(0, 100))
p.border_fill_color = 'black'
p.background_fill_color = 'black'
p.outline_line_color = None
p.grid.grid_line_color = None

# add a text renderer to out plot (no data yet)
r = p.text(x=[], y=[], text=[], text_color=[], text_font_size="20pt",
           text_baseline="middle", text_align="center")

session = push_session(curdoc())


data = dict(
        dates=[date(2014, 3, i+1) for i in range(10)],
        downloads=[randint(0, 100) for i in range(10)],
    )
source = ColumnDataSource(data)

columns = [
        TableColumn(field="dates", title="Date", formatter=DateFormatter()),
        TableColumn(field="downloads", title="Downloads"),
    ]
data_table = DataTable(source=source, columns=columns, width=400, height=280)

curdoc().add_root(vform(data_table))

session.show()

