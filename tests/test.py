myDict1 = {'1': ('3', '2'), '3': ('2', '1'), '2': {'1': 5, '3': 7, '5': 9, '6': {'p': 2, 'q': 4, '5': {'1': 3}}}}
myDict2 = {'1': ['5', '2'], '5': ('2', '4'), '2': {'2': 6, '4': 8, '6': {'1': 2, '3': 4}}}

def update_dict_simple(myDict1, myDict2):
   for k, v in myDict2.items():
      if isinstance(v, dict):
         myDict1[k] = {**myDict1.get(k, ()), **v}
      elif isinstance(v, tuple):
         myDict1[k] = myDict1.get(k, ()) + v
      elif isinstance(v, list):
         if isinstance(myDict1[k], tuple):
            myDict1[k] = [*list(myDict1.get(k, ())), *v]
         elif isinstance(myDict1[k], list):
            myDict1[k] = [*myDict1.get(k, ()), *v]
   return myDict1

def update_dict(del_dict: dict, temp_dict: dict, replace: bool = True, add_val: bool = False,\
                sep: str = "_", suffix: str = "0", prefix: str = "")-> dict:
   for k, v in temp_dict.items():
      if not isinstance(k, str):
         k = str(k)
      if isinstance(v, dict):
         del_dict[k] = {**del_dict.get(k, ()), **update_dict(del_dict[k], v)}
      elif isinstance(v, tuple):
         del_dict[k] = del_dict.get(k, ()) + v
      elif isinstance(v, list):
         if isinstance(del_dict[k], tuple):
            del_dict[k] = [*list(del_dict.get(k, ())), *v]
         elif isinstance(del_dict[k], list):
            del_dict[k] = [*del_dict.get(k, ()), *v]
      else:
         if k in list(del_dict):
            print(f"TEMP KEY IN ORIGINAL DICT --- FOR SINGLE VALUE: {k}")
            if replace and not add_val:
               del_dict[k] = v
            elif not replace and add_val:
               del_dict[prefix + k + sep + suffix] = v
            elif not replace and not add_val:
               pass
         else:
            del_dict[k] = v
   return del_dict

print(f"\n\nOld: {myDict1}")
myDict3 = update_dict_simple(myDict1, myDict2)
print(f"\n\nNew: {myDict3}")

print(f"\n\nOld: {myDict1}")
myDict1 = update_dict(myDict1, myDict2)
print(f"\n\nNew: {myDict1}")

"""def _get_all_paths(dict_, temp = []):
   for k, v in dict_.items():
      temp.append(k)
      if isinstance(v, dict):
         item = _get_all_paths(v, temp)
         if item is not None:
            return del_hist
      # Returning String Combination
      strs = ""
      del_hist.append(strs.join(t + "_" for i, t in enumerate(temp) if i != len(temp)-1)[:-1])
   return del_hist"""