#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 11:00:43 2018

@author: ehsan
"""

from matplotlib import collections  as mc
import pylab as pl
from matplotlib import colors as mcolors
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

booking = pd.read_csv('room_assignment.csv')

booking.arrival_date = pd.to_datetime(booking.arrival_date).dt.date
booking.departure_date = pd.to_datetime(booking.departure_date).dt.date


booking['arrival_number'] = matplotlib.dates.date2num(booking.arrival_date)
booking['departure_number'] = matplotlib.dates.date2num(booking.departure_date)


def visualize_bookings_with_not_move(bookings,color):
    lines = []
    colors = []
    for _,booking in bookings.iterrows():
        lines.append([(matplotlib.dates.date2num(booking.arrival_date),booking.unit_number),
                      (matplotlib.dates.date2num(booking.departure_date),booking.unit_number)])
        colors.append(['blue','red'][booking.not_move])
    
    lc = mc.LineCollection(lines[:-1],linewidths=3,colors=colors)
    return lc


def visualize_bookings(bookings,color):
    lines = []

    for _,booking in bookings.iterrows():
        lines.append([(matplotlib.dates.date2num(booking.arrival_date),booking.unit_number),
                      (matplotlib.dates.date2num(booking.departure_date),booking.unit_number)])
    
    
    lc = mc.LineCollection(lines[:-1],linewidths=3,color=color)
    return lc



#fig, ax = pl.subplots()
#plt.xlabel('time')
#plt.ylabel('units')
#ax.add_collection(lc)
#ax.autoscale()
#ax.margins(0.1)
#plt.show()