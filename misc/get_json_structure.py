# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 22:25:19 2024

@author: Gabin
"""

def get_json_structure(data, dictionary=None):
    if dictionary is None:
        dictionary = {}
    if isinstance(data, dict):
        for key, value in data.items():
            dictionary[key] = get_json_structure(value)
    elif isinstance(data, list):
        if data:
            dictionary = [get_json_structure(data[0])]
        else:
            dictionary = []
    else:
        dictionary = type(data).__name__
    return dictionary
