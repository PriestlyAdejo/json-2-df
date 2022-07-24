# ---------------------------------------------------------------------
# JSON normalization routines
from __future__ import annotations

from collections import (
    abc,
    defaultdict,
    OrderedDict
)
import copy
from typing import (
    Any,
    DefaultDict,
    Iterable,
)

import numpy as np

from pandas._libs.writers import convert_json_to_lines
from pandas._typing import * # Other imports raise errors
from pandas.util._decorators import deprecate

import pandas as pd
from pandas import DataFrame

import numpy as np
import json
# ---------------------------------------------------------------------

"""
This function is heavily based on the nested_to_record() dictionary normalising function inside 
pandas._normalize().
"""

# Helper Functions
def get_dict_path(kh: dict, ignore: dict, ig_key: str)-> dict[str, Any]:
   """
   Helper Function to get all parent keys up to the point
   of deletion in a dictionary. Reorginises current disorganised path dict (kh)
   and checks for ignore values on the same and consecutive levels.
   """
   new_keys = []
   new_lvls = []
   ignore_count = 0

   set_lvls = set(list(kh.values()))
   for lvl in set_lvls:
      idxs = [(idx_in_dict, key) for idx_in_dict, (key, val) in enumerate(list(kh.items())) if val == lvl]
      idx_ig_key = [i for i, key in idxs if key == ig_key]
      idx_ig_key = idx_ig_key[0] if len(idx_ig_key) >= 1 else idx_ig_key
      temp_kh = {k: v for k, v in list(kh.items()) if v == lvl}
      
      # Finds num ignore cols in one level
      ig_sl_count = 0
      for key in list(temp_kh):
         if key in ignore["cols"]:
            ig_sl_count += 1
            if ig_sl_count > 1:
               break
      
      if not ig_sl_count > 1:
         # If cols to ignore exist on consecutive lvls, keep the first col
         new_keys.append(list(kh)[idxs[-1][0]])
         new_lvls.append(list(kh.values())[idxs[-1][0]])
         if list(kh)[idxs[-1][0]] in ignore["cols"]:
            ignore_count += 1
         if ignore_count > 1:
            # Prioritises first ignore seen for consecutive lvls
            new_lvls = new_lvls[:-1] 
            new_keys = new_keys[:-1]
      else:
         new_keys.append(list(kh)[idx_ig_key])
         new_lvls.append(list(kh.values())[idx_ig_key])

   # Resetting history
   kh = {} 
   kh = {k: v for k, v in zip(new_keys, new_lvls)}
   return kh

def get_locs(kh: dict)-> list:
   """
   Function to get strings of all parent keys up to del point
   """
   return [[key for key in list(kh)],\
            [lvl for lvl in list(kh.values())]] 

def get_multi_dict(splits: list, val: Any)-> dict[str, Any]:
   """
   Function to recursively create multilevel dicitonary
   from a list of key locations. Takes desired value to be put at that location.
   """
   if len(splits) == 0:
      return val
   elif isinstance(splits, list):
      first_value = splits[0] 
      splits_dict = {first_value : get_multi_dict(splits[1:], val)}
      return splits_dict

# Helper Functions to separte each deleted val into a separted dictionary, 
# with user chosen lvls for names. These operate like a pivot table (in sql).
def locs_to_val(d_dict: dict, loc_arr: list, lvl: int = 0)-> Any:
   """
   Uses recursion to grab value of dictionary given a list, its the reverse of
   get_multi_dict(), and gets a value rather than creating a dictionary.
   """
   if isinstance(d_dict, dict) and lvl != len(loc_arr)-1:
      for key, val in d_dict.items():
         if isinstance(val, dict):
            if lvl < len(loc_arr):
               if key == loc_arr[lvl]:
                  new_val = val
                  return locs_to_val(new_val, loc_arr, lvl + 1)
   elif isinstance(d_dict, dict) and lvl == len(loc_arr)-1:
      return d_dict[loc_arr[-1]]
   return d_dict[loc_arr[-1]]

def update_dict(del_dict: dict, temp_dict: dict, replace: bool = True, add_val: bool = False,\
            sep: str = "_", suffix: str = "0", prefix: str = "")-> dict[str, Any]:
   """
   Function that updates multilevel dictionary without resetting values. Much like an
   implementation of the list.append() function but for dictionaries. User can
   specify whether to replace, ignore or append certain additions.
   """
   if isinstance(temp_dict, dict):
      for k, v in temp_dict.items():
         if not isinstance(k, str):
            k = str(k)
         if isinstance(v, dict) and isinstance(del_dict.get(k, {}), dict):
            del_dict[k] = {**del_dict.get(k, {}), **update_dict(del_dict.get(k, {}), v)}
         elif isinstance(v, tuple):
            del_dict[k] = del_dict.get(k, ()) + v
         elif isinstance(v, list):
            if isinstance(del_dict.get(k, ()), tuple):
               del_dict[k] = [*list(del_dict.get(k, ())), *v]
            elif isinstance(del_dict.get(k, []), list):
               del_dict[k] = [*del_dict.get(k, []), *v]
         else:
            if k in list(del_dict):
               if replace and not add_val:
                  del_dict[k] = v
               elif not replace and add_val:
                  del_dict[prefix + k + sep + suffix] = v
               elif not replace and not add_val:
                  pass
            else:
               del_dict[k] = v
   return del_dict

def get_del(ignore_col: str, del_dict: dict, dels: list, max_level: int or None = None,\
   ignore_loc: bool = False)-> dict[str, Any]:
   """
   Helper Function to create a "pivot_table" (see earlier) of the delete values.
   It uses recursion in locs_to_val() to search for a value to put in this new
   ordered dictionary. The result is appended to the main dicitonary containing all the pivots.
   It can either be used with a list of location lists or a sinlge location list (to reduce
   time and soze complexities for large dictionaries).
   """
   # max_level always starts from 0
   dict_cols = {}
   dels = np.array(dels)

   if len(dels.shape) == 2: # If a list of paths is passed
      for idx in range(len(dels)):
         if ignore_col in dels[idx]:
            if len(dels[idx]) >= (max_level):
               dict_cols[str(dels[idx][max_level])] = locs_to_val(del_dict, dels[idx])
            elif max_level == None:
               # If max_level is none, use lowest level available
               dict_cols[str(dels[idx][0])] = locs_to_val(del_dict, dels[idx])
            else:
               if not ignore_loc:
                  # If max_level is greater than len(loc_lvls) use level at maximum idx
                  dict_cols[str(dels[idx][-1])] = locs_to_val(del_dict, dels[idx])
               else:
                  # Ignore a column if incorrect max_level supplied
                  pass
   elif len(dels.shape) == 1: # If a single path is passed
      if ignore_col in dels:
         if len(dels) >= (max_level):
            dict_cols[str(dels[max_level])] = locs_to_val(del_dict, dels)
         elif max_level == None:
            # If max_level is none, use lowest level available
            dict_cols[str(dels[0])] = locs_to_val(del_dict, dels)
         else:
            if not ignore_loc:
               # If max_level is greater than len(loc_lvls) use level at maximum idx
               dict_cols[str(dels[-1])] = locs_to_val(del_dict, dels)
            else:
               # Ignore a column if incorrect max_level supplied
               pass
   return dict_cols

# Returns normalised_json, (unordered_dict_of_deleted_vals, locations_of_deleted_vals,
# pivot_tables_of_deleted_vals, path_idx)
def nested_ignore_cols_to_record(
   ds,
   prefix: str = "",
   sep: str = ".",
   level: int = 0,
   max_level: int or None = None,
   keys_hist: dict = {},
   del_dict: dict = {},
   dels: list = [],
   pivot_dels: dict = {},
   return_dels: bool = False,
   path_idx: int = 0,
   first_update: bool = True,
   ignore: dict = {"cols": None,\
                   "name_lvls": None}
):
   """
    A more commplex version of nested_to_record(), user can pass in columns to
    ignore in the input dictionary (before flattening).

    Can also return the deleted values so they can be used for other cases.

    If no column to be ignored is specified, function just performs the 
    un-edited nested_to_record().

    Only truly deletes values if no max_level is specified (max_level == None).

    Parameters
    ----------
    ds : dict or list of dicts
    prefix: the prefix, optional, default: ""
    sep : str, default '.'
        Nested records will generate names separated by sep,
        e.g., for sep='.', { 'foo' : { 'bar' : 0 } } -> foo.bar
    level: int, optional, default: 0
        The number of levels in the json string.
    max_level: int, optional, default: None
        The max depth to normalize.
    keys_hist: dict, optional, default: {}
        This saves the path to a value to be deleted, gets reset
        after the value gets deleted.
    del_dict: dict, optional, default: {}
        This saves the values deleted from the original data in their
        own dictionary. THIS DICTIONARY IS UNORDERED.
    dels: list, optional, default: []
        This is an extension of del_dict, saves the locations of each deleted
        value in an array as well as their levels leading up to the deleted value.
    pivot_dels: dict, optional, default: {}
        This is the ordered version of del_dict. The top level keys are the deleted value
        keys/columns and the values are dictionaries with user-specified "name_lvl" keys
        in ignore (see below).
    return_dels: bool, optional, default: True
        If true, returns the deleted values, as well as the flattened input dictionary,
        in the format: flattened_dict, (del_dict, dels, pivot_dels, path_idx)
    path_idx: int, default: 0
        Internal variable used to find the updated running idx of values in dels.
    ignore: dict, optional, default: {"cols": None, "name_lvls": None}
        This contains the columns to be deleted and the desired levels (in the array of the
        locations of deleted values) to be used as keys in the level below the top level of
        the ordered pivot_dels dictionary (see above).

    .. versionadded:: x.xx.x --- Please fill in

    Returns
    -------
    1. d - dict or list of dicts, matching `ds`
    or
    2. (1.) and del_tuple - tuple of (del_dict, dels, pivot_dels, path_idx) (see above)

    Example (1.)
    --------
    nested_ignore_cols_to_record(
      dict(flat1=1, dict1=dict(c=1, d=2), nested=dict(e=dict(c=1, d=2), d=2))

            {
        'flat1': 1, 
        'dict1.c': 1, 
        'dict1.d': 2, 
        'nested.e.c': 1, 
        'nested.e.d': 2, 
        'nested.d': 2
        }

    Example (2.)
    --------
    nested_ignore_cols_to_record(
         dict(flat1=1, dict1=dict(c=1, d=2), nested=dict(e=dict(c=1, d=2), d=2),
                ignore = {"cols": ["e"], "name_lvls": None},
                return_dels = True)

       returns: NORMALISED DICT WITH 'e' col ignored:
               {'flat1': 1, 'dict1.c': 1, 'dict1.d': 2, 'nested.d': 2},

                FULL_DELS_TUPLE:
               ({'nested': {'e': {'c': 1, 'd': 2}}}, [['nested', 'e']], {'e': {'nested': {'c': 1, 'd': 2}}}, 1)
               
               FULL_DELS_TUPLE[0] = del_dict
               FULL_DELS_TUPLE[1] = dels
               FULL_DELS_TUPLE[2] = pivot_dels
               FULL_DELS_TUPLE[3] = path_idx
   """
   singleton = False
   if isinstance(ds, dict):
      ds = [ds]
      singleton = True
   new_ds = []
   for d in ds:
      new_d = copy.deepcopy(d)
      same_key = 0
      path_arr = []
      for k, v in d.items():
         # each key gets renamed with prefix
         if not isinstance(k, str):
            k = str(k)

         if len(list(keys_hist)) > 0:
            # Checking for same consecutive keys
            if k == list(keys_hist)[-1]:
               same_key += 1
               keys_hist[k + "_" + str(same_key)] = level
            else:
               keys_hist[k] = level

         if ignore["cols"] is None:
            if level == 0:
               newkey = k
               keys_hist[k] = level
            else:
               newkey = prefix + sep + k
         elif k not in ignore["cols"]:
            if level == 0:
               newkey = k
               keys_hist[k] = level
            else:
               newkey = prefix + sep + k
         elif k in ignore["cols"]:
            if first_update == True:
               del_dict = {}
               pivot_dels = {}
               dels = []
               path_arr = []

               keys_hist = get_dict_path(keys_hist, ignore, k)
               locs, _ = get_locs(keys_hist)
               dels.append(locs)
               path_arr.append((locs, path_idx))
               path_idx += 1
               first_update = False
            else:
               keys_hist = get_dict_path(keys_hist, ignore, k)
               locs, _ = get_locs(keys_hist)
               dels.append(locs)
               path_arr.append((locs, path_idx))
               path_idx += 1
            continue

         # flatten if type is dict and
         # current dict level < maximum level provided and
         # only dicts gets recurse-flattened
         # only at level>1 do we rename the rest of the keys
         if not isinstance(v, dict) or (
            max_level is not None and level >= max_level
         ):
            if level != 0:  # so we skip copying for top level, common case
               v = new_d.pop(k)
               new_d[newkey] = v
            continue
         else:
            v = new_d.pop(k)
            keys_hist = get_dict_path(keys_hist, ignore, k)

            if return_dels == True:
               nest_dict, d_tuple = \
               nested_ignore_cols_to_record(v, newkey, sep, level + 1, max_level,\
                           keys_hist, del_dict, dels, pivot_dels, return_dels, path_idx, first_update = first_update, ignore = ignore)
               del_dict = d_tuple[0]
               dels = d_tuple[1]
               pivot_dels = d_tuple[2]
               path_idx = d_tuple[3]
               
               # Updating first_update
               for item in d_tuple:
                  if isinstance(item, dict):
                     if len(item.items()) > 0:
                        first_update = False
                  elif isinstance(item, list):
                     if len(item) > 0:
                        first_update = False
                  if isinstance(item, int) or isinstance(item, float):
                     if item > 0:
                        first_update = False
                  break
            else:
               nest_dict = \
               nested_ignore_cols_to_record(v, newkey, sep, level + 1, max_level,\
                           keys_hist, del_dict, dels, pivot_dels, return_dels, path_idx, first_update, ignore = ignore)
            new_d.update(nest_dict)
      new_ds.append(new_d)
   
   if singleton:
      if (ignore["cols"] is not None and len(path_arr) > 0
          and ignore["name_lvls"] is not None
          ):
         # Resetting all dels_tup for repeated implementations
         del_tups = [t[0] for t in path_arr]
         temp_d = {}
         for del_tup in del_tups:
            # Assuming del_val is last in each del_tup array
            last_val = del_tup[-1] 
            if last_val in list(new_ds[0]) and last_val in ignore["cols"]:
               ig_ml_tup = [(ig, ml) for ig, ml in zip(ignore["cols"], ignore["name_lvls"])\
                           if ig == last_val]
               ignore_col = ig_ml_tup[0][0]
               ml = ig_ml_tup[0][1]
               val = new_ds[0].pop(last_val)
               temp = get_multi_dict(del_tup, val)
               del_dict = update_dict(del_dict, temp)

               if ignore["name_lvls"] is not None:
                  temp_d[ignore_col] = get_del(ignore_col, del_dict, del_tup, ml)
               else:
                  temp_d[ignore_col] = get_del(ignore_col, del_dict, del_tup, 0)

         pivot_dels = update_dict(pivot_dels, temp_d)

      if return_dels == True: # User defined
         return new_ds[0], (del_dict, dels, pivot_dels, path_idx)
      return new_ds[0]

   if (ignore["cols"] is not None and len(path_arr) > 0
       and ignore["name_lvls"] is not None
          ):
      del_tups = [t[0] for t in path_arr]
      temp_d = {}
      for del_tup in del_tups:
         # Assuming del_val is last in each del_tup array
         last_val = del_tup[-1] 
         if last_val in list(new_ds) and last_val in ignore["cols"]:
            ig_ml_tup = [(ig, ml) for ig, ml in zip(ignore["cols"], ignore["name_lvls"])\
                           if ig == last_val]
            ignore_col = ig_ml_tup[0][0]
            ml = ig_ml_tup[0][1]
            val = new_ds.pop(last_val)
            temp = get_multi_dict(del_tup, val)
            del_dict = update_dict(del_dict, temp)

            if ignore["name_lvls"] is not None:
               temp_d[ignore_col] = get_del(ignore_col, del_dict, del_tup, ml)
            else:
               temp_d[ignore_col] = get_del(ignore_col, del_dict, del_tup, 0)

      pivot_dels = update_dict(pivot_dels, temp_d)

   if return_dels == True: # User defined
      return new_ds, (del_dict, dels, pivot_dels, path_idx)
   return new_ds

