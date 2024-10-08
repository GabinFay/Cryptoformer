#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 00:59:18 2024

@author: gabinfay
"""

import datetime, os; print(f"{datetime.datetime.now()} running on {os.path.basename(__file__)}")

from coingecko import hourly_sheets_update_bins, hourly_sheets_update_5000_flat, hourly_sheets_update_small_mcap, hourly_sheets_update_big_mcap
hourly_sheets_update_small_mcap()
hourly_sheets_update_big_mcap()
hourly_sheets_update_5000_flat()
hourly_sheets_update_bins()